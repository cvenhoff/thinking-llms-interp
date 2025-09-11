import math
import random
import gc
from contextlib import contextmanager
from typing import List, Tuple, Callable, Optional, Union

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# =============================================================
# 1.  Context‑manager helpers
# =============================================================

@contextmanager
def hf_hooks_contextmanager(model, hook_infos: List[Tuple[int, Callable]]):
    """Attach *forward‑pre* hooks to a HF decoder‑only model inside a 'with' block."""
    hooks = [model.model.layers[layer].register_forward_pre_hook(hook) for layer, hook in hook_infos]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# =============================================================
# 2.  Steering‑hook factory
# =============================================================

def make_steering_hook_hf(vector: torch.Tensor,
                           projection_clamp: bool = False,
                           token: Optional[Union[int, slice]] = None):
    """Return a `forward_pre_hook` that injects the steering vector *in‑place*.

    Args
    ----
    vector : torch.Tensor
        The learned steering vector (1‑D, d_model).
    projection_clamp : bool
        If *True* use the projection‑clamp transform
            x  ->  x - (x·v / ‖v‖²) v  +  v
        instead of the simple additive tweak x += v.
    token : int | slice | None
        Token position(s) to which the steering applies.  Default = all tokens.
    """

    if token is None:
        token = slice(None)

    v = vector  # close‑over a pointer; we will deref in‑place so it stays up‑to‑date

    def hook_fn(_module, args):
        (x,) = args  # NOTE: forward_pre hooks get *one* tuple‑d arg list
        v_local = v.to(x)

        # Select the slice we manipulate
        x_slice = x[:, token]

        if projection_clamp:
            # x_slice <- proj‑clamp(x_slice, v)
            coef = (x_slice @ v_local) / (v_local.norm() ** 2)
            x[:, token] = x_slice - coef.unsqueeze(-1) * v_local + v_local
        else:
            x[:, token] = x_slice + v_local

        return (x,)  # MUST return a tuple!

    return hook_fn


# =============================================================
# 2b.  LoRA steering‑hook factory
# =============================================================

def make_resid_lora_hook_hf(A: torch.Tensor,
                      B: torch.Tensor,
                      alpha: torch.Tensor,
                      token: Optional[Union[int, slice]] = None):
    if token is None:
        token = slice(None)

    def hook_fn(_module, args):
        (x,) = args
        d_model = x.shape[-1]
        assert A.shape[0] == d_model and B.shape[1] == d_model, "A and B must map d_model -> r -> d_model"
        Al = A.to(x)
        Bl = B.to(x)
        al = alpha.to(x)
        xs = x[:, token]
        x[:, token] = xs + al * ((xs @ Al) @ Bl)
        return (x,)

    return hook_fn

# =============================================================
# 3.  Generic cosine‑anneal scheduler (optionally disabled)
# =============================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return cosine_decay * (1 - min_lr / base_lr) + min_lr / base_lr

    return LambdaLR(optimizer, lr_lambda)


# =============================================================
# 4.  Optimisation loop
# =============================================================

def optimize_vector_simple(
    model,
    tokenizer,
    prompts,
    target_completions,
    layer: int,
    *,
    lr: float = 0.01,
    max_iters: int = 50,
    optim_minibatch_size: int = 32,
    base_gen_minibatch_size: int = 32,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
    coldness: float = 0.7,
    grad_clip: Optional[float] = None,
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 1e-4,
    max_norm: Optional[float] = None,
    starting_norm: float = 1.0,
    debug: bool = False,
    return_info: bool = True,
    return_loss_history: bool = False,
    steering_token_window: Optional[int] = None,
    projection_clamp: bool = False,
    wandb_run=None,
    static_vectors: Optional[list[torch.Tensor]] = None,
    include_base_objective: bool = False,
    base_loss_weight: float = 1.0,
    eps: float = 1e-6,
    eval_prompts: Optional[list[str]] = None,
    eval_target_completions: Optional[list[str]] = None,
    # Unified steering controls
    steering_type: str = "linear",  # "linear" or "resid_lora"
    rank: int = 1,
):
    """One‑stop promotion‑steering optimiser matching the paper’s details.

    Supports two parameterisations under one API:
      - steering_type == "linear": train a 1-D vector added to hidden states
      - steering_type == "resid_lora":   train a rank-r residual LoRA: x += alpha * ((x @ A) @ B)
    """

    # ---------------- Pre‑flight checks ---------------- #
    if len(prompts) != len(target_completions):
        raise ValueError("Prompts and completions length mismatch.")
    # Evaluation logic removed; optimisation now relies only on training loss.

    # ---------------- Parameter initialisation ----------- #
    d_model = model.config.hidden_size
    if steering_type == "linear":
        vector = torch.randn(d_model, device=model.device)
        vector = starting_norm * vector / vector.norm()
        vector.requires_grad_(True)
        params_to_opt = [vector]
    elif steering_type == "resid_lora":
        assert rank >= 1 and rank <= d_model, "Invalid LoRA rank"
        A = torch.randn(d_model, rank, device=model.device) / math.sqrt(d_model)
        B = torch.randn(rank, d_model, device=model.device) / math.sqrt(d_model)
        alpha_param = torch.tensor(1.0, device=model.device)
        A.requires_grad_(True)
        B.requires_grad_(True)
        alpha_param.requires_grad_(True)
        params_to_opt = [A, B, alpha_param]
    else:
        raise ValueError("steering_type must be 'linear' or 'resid_lora'")

    # ---------------- Tokenisation helpers -------------- #
    def tok_batch(strs):
        toks, lengths = [], []
        for s in strs:
            t = tokenizer(s, return_tensors="pt").input_ids[0]
            toks.append(t)
            lengths.append(t.size(0))
        return toks, lengths

    prompt_tokens, prompt_lens = tok_batch(prompts)
    target_tokens, target_lens = tok_batch(target_completions)

    # ---- Optional evaluation tokenisation ----
    have_eval = eval_prompts is not None and eval_target_completions is not None and len(eval_prompts) > 0
    if have_eval:
        eval_prompt_tokens, eval_prompt_lens = tok_batch(eval_prompts)
        eval_target_tokens, eval_target_lens = tok_batch(eval_target_completions)

    # ---- Precompute unsteered base completions (used as "src" completions in the objective) ----
    base_completions: list[str] = []
    if include_base_objective:
        descr = "Precomputing base completions" if static_vectors is None else "Precomputing base completions with static vectors"
        
        # Batching for efficiency
        for i in tqdm(range(0, len(prompts), base_gen_minibatch_size), desc=descr):
            batch_prompts = prompts[i:i + base_gen_minibatch_size]
            batch_target_lens = target_lens[i:i + base_gen_minibatch_size]

            max_new = max(1, max(tl.item() if isinstance(tl, torch.Tensor) else int(tl) for tl in batch_target_lens))

            inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=False).to(model.device)
            
            hooks = []
            if static_vectors is not None:
                clean_static_vectors = [sv for sv in static_vectors if sv is not None]
                if clean_static_vectors:
                    prompt_lens_tensor = inputs['attention_mask'].sum(dim=1)
                    def static_vec_batch_hook(_m, args, p_lens=prompt_lens_tensor):
                        (x,) = args
                        for row in range(x.shape[0]):
                            p_len = p_lens[row].item()
                            sl = slice(p_len, x.shape[1])
                            for sv in clean_static_vectors:
                                x[row, sl] += sv.to(x)
                        return (x,)
                    hooks.append((layer, static_vec_batch_hook))

            with torch.no_grad():
                generation_kwargs = dict(
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.eos_token_id,
                    suppress_tokens=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
                )
                if hooks:
                    with hf_hooks_contextmanager(model, hooks):
                        gen = model.generate(**inputs, **generation_kwargs)
                else:
                    gen = model.generate(**inputs, **generation_kwargs)
            
            decoded_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for j, txt in enumerate(decoded_texts):
                original_prompt = batch_prompts[j]
                base_completions.append(txt[len(original_prompt):])
    else:
        base_completions = [""] * len(prompts)

    base_tokens, base_lens = tok_batch(base_completions)

    # Precompute base completions for eval set if provided and base objective enabled
    if have_eval:
        if include_base_objective:
            descr_eval = "Precomputing base completions (eval)" if static_vectors is None else "Precomputing base completions with static vectors (eval)"
            base_completions_eval: list[str] = []
            for i in tqdm(range(0, len(eval_prompts), base_gen_minibatch_size), desc=descr_eval):
                batch_prompts = eval_prompts[i:i + base_gen_minibatch_size]
                batch_target_lens = eval_target_lens[i:i + base_gen_minibatch_size]
                
                max_new = max(1, max(tl.item() if isinstance(tl, torch.Tensor) else int(tl) for tl in batch_target_lens))

                inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=False).to(model.device)
                
                hooks = []
                if static_vectors is not None:
                    clean_static_vectors = [sv for sv in static_vectors if sv is not None]
                    if clean_static_vectors:
                        prompt_lens_tensor = inputs['attention_mask'].sum(dim=1)
                        def static_vec_batch_hook_eval(_m, args, p_lens=prompt_lens_tensor):
                            (x,) = args
                            for row in range(x.shape[0]):
                                p_len = p_lens[row].item()
                                sl = slice(p_len, x.shape[1])
                                for sv in clean_static_vectors:
                                    x[row, sl] += sv.to(x)
                            return (x,)
                        hooks.append((layer, static_vec_batch_hook_eval))

                with torch.no_grad():
                    generation_kwargs = dict(
                        max_new_tokens=max_new,
                        pad_token_id=tokenizer.eos_token_id,
                        suppress_tokens=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
                    )
                    if hooks:
                        with hf_hooks_contextmanager(model, hooks):
                            gen = model.generate(**inputs, **generation_kwargs)
                    else:
                        gen = model.generate(**inputs, **generation_kwargs)
                
                decoded_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for j, txt in enumerate(decoded_texts):
                    original_prompt = batch_prompts[j]
                    base_completions_eval.append(txt[len(original_prompt):])
            
            base_tokens_eval, base_lens_eval = tok_batch(base_completions_eval)
        else:
            base_tokens_eval = [torch.tensor([], device=model.device)] * len(eval_prompts)  # type: ignore
            base_lens_eval = [0] * len(eval_prompts)  # type: ignore
    else:
        base_tokens_eval, base_lens_eval = None, None

    # Evaluation dataset and metrics have been removed.
    eval_hist = []

    # ---------------- Optimiser & LR schedule ----------- #
    optim = torch.optim.Adam(params_to_opt, lr=lr)

    # Ensure static_vectors is a list of tensors on correct device
    if static_vectors is None:
        static_vectors_local: list = []
    else:
        static_vectors_local = [sv.to(model.device).detach() if isinstance(sv, torch.Tensor) else sv for sv in static_vectors]

    if steering_type == "resid_lora" and static_vectors_local:
        # Expect dicts with A,B,alpha for LoRA static adapters
        for sv in static_vectors_local:
            assert isinstance(sv, dict) and all(k in sv for k in ("A","B","alpha")), "LoRA static_vectors must be dicts with A,B,alpha"

    # ---- learning-rate schedule ----
    # We want the scheduler to advance *once per optimiser step* (i.e. per minibatch).
    # Hence, total training steps = max_iters * num_batches_per_epoch.
    num_batches_per_epoch = (len(prompts) + optim_minibatch_size - 1) // optim_minibatch_size
    total_training_steps = max_iters * num_batches_per_epoch

    sched = (
        get_cosine_schedule_with_warmup(optim, warmup_steps, total_training_steps, min_lr)
        if total_training_steps
        else None
    )

    # ---------------- Training loop --------------------- #
    best_vec, best_lora, best_loss = None, None, float("inf")
    loss_hist = []
    patience = 0

    pbar = tqdm(range(max_iters), desc="")

    for step in pbar:
        # ----- minibatch shuffling -----
        idxs = list(range(len(prompts)))
        random.shuffle(idxs)

        running_loss, batches = 0.0, 0
        batch_pbar = tqdm(range(0, len(idxs), optim_minibatch_size), desc=f"batches {step+1}/{max_iters}", position=1, leave=False)
        for bs in batch_pbar:
            batch = idxs[bs : bs + optim_minibatch_size]

            # Build *right‑padded* batch tensors
            seqs = [torch.cat([prompt_tokens[i], target_tokens[i]]) for i in batch]
            max_len = max(s.size(0) for s in seqs)

            pad_id = tokenizer.pad_token_id
            input_ids = torch.full((len(batch), max_len), pad_id, device=model.device)
            attn_mask = torch.zeros_like(input_ids)

            steering_slices = []
            for row, (i, seq) in enumerate(zip(batch, seqs)):
                L = seq.size(0)
                input_ids[row, :L] = seq  # right‑pad
                attn_mask[row, :L] = 1

                start = prompt_lens[i] if steering_token_window is None else (
                    prompt_lens[i] + max(0, target_lens[i] - steering_token_window)
                )
                steering_slices.append(slice(start, L))

            # Hook for this batch
            if steering_type == "resid_lora":
                def batch_hook(_m, args, slices=steering_slices, stat_vecs=static_vectors_local):
                    (x,) = args
                    v_local = vector.to(x)
                    stat_vecs_on_device = [sv.to(x.device) for sv in stat_vecs]
                    for row, sl in enumerate(slices):
                        if projection_clamp:
                            seg = x[row, sl]
                            coef = (seg @ v_local) / (v_local.norm() ** 2)
                            x[row, sl] = seg - coef.unsqueeze(-1) * v_local + v_local
                        else:
                            x[row, sl] += v_local
                        if stat_vecs_on_device:
                            for sv in stat_vecs_on_device:
                                x[row, sl] += sv
                    return (x,)
            else:
                def batch_hook(_m, args, slices=steering_slices, A_param=A, B_param=B, alpha_p=alpha_param, stat_loras=static_vectors_local):
                    (x,) = args
                    Al = A_param.to(x)
                    Bl = B_param.to(x)
                    al = alpha_p.to(x)
                    for row, sl in enumerate(slices):
                        seg = x[row, sl]
                        x[row, sl] = seg + al * ((seg @ Al) @ Bl)
                        if stat_loras:
                            for l in stat_loras:
                                Al_s = l['A'].to(x)
                                Bl_s = l['B'].to(x)
                                al_s = l['alpha'].to(x)
                                x[row, sl] = x[row, sl] + al_s * ((x[row, sl] @ Al_s) @ Bl_s)
                    return (x,)

            with hf_hooks_contextmanager(model, [(layer, batch_hook)]):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits * coldness

            # Cross‑entropy on *targets only*
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Mask out padding and prompt tokens
            target_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            for row, (i, sl) in enumerate(zip(batch, steering_slices)):
                tgt_start = sl.start - 1  # shift by one because of next‑token prediction
                tgt_end = steering_slices[row].stop - 1 if steering_slices[row].stop is not None else shift_labels.size(1)
                target_mask[row, tgt_start:tgt_end] = True

            active_logits = shift_logits[target_mask]
            active_labels = shift_labels[target_mask]
            ce_target = torch.nn.functional.cross_entropy(active_logits, active_labels, reduction="mean") / math.log(10)

            # ---- Base (unsteered) completion objective: -log(1 - p_base) on base tokens ----
            base_loss = 0.0
            if include_base_objective:
                base_seqs = [torch.cat([prompt_tokens[i], base_tokens[i]]) for i in batch]
                max_len_b = max(s.size(0) for s in base_seqs)

                input_ids_b = torch.full((len(batch), max_len_b), pad_id, device=model.device)
                attn_mask_b = torch.zeros_like(input_ids_b)

                steering_slices_b = []
                for row, (i, seq) in enumerate(zip(batch, base_seqs)):
                    Lb = seq.size(0)
                    input_ids_b[row, :Lb] = seq
                    attn_mask_b[row, :Lb] = 1

                    start_b = prompt_lens[i] if steering_token_window is None else (
                        prompt_lens[i] + max(0, (base_lens[i] if isinstance(base_lens[i], int) else int(base_lens[i])) - steering_token_window)
                    )
                    steering_slices_b.append(slice(start_b, Lb))

                if steering_type == "linear":
                    def batch_hook_base(_m, args, slices=steering_slices_b, stat_vecs=static_vectors_local):
                        (x,) = args
                        v_local = vector.to(x)
                        stat_vecs_on_device = [sv.to(x.device) for sv in stat_vecs]
                        for row, sl in enumerate(slices):
                            if projection_clamp:
                                seg = x[row, sl]
                                coef = (seg @ v_local) / (v_local.norm() ** 2)
                                x[row, sl] = seg - coef.unsqueeze(-1) * v_local + v_local
                            else:
                                x[row, sl] += v_local
                            if stat_vecs_on_device:
                                for sv in stat_vecs_on_device:
                                    x[row, sl] += sv
                        return (x,)
                elif steering_type == "resid_lora":
                    def batch_hook_base(_m, args, slices=steering_slices_b, A_param=A, B_param=B, alpha_p=alpha_param, stat_loras=static_vectors_local):
                        (x,) = args
                        Al = A_param.to(x)
                        Bl = B_param.to(x)
                        al = alpha_p.to(x)
                        for row, sl in enumerate(slices):
                            seg = x[row, sl]
                            x[row, sl] = seg + al * ((seg @ Al) @ Bl)
                            if stat_loras:
                                for l in stat_loras:
                                    Al_s = l['A'].to(x)
                                    Bl_s = l['B'].to(x)
                                    al_s = l['alpha'].to(x)
                                    x[row, sl] = x[row, sl] + al_s * ((x[row, sl] @ Al_s) @ Bl_s)
                        return (x,)
                else:
                    raise ValueError(f"Unknown steering_type: {steering_type}")

                with hf_hooks_contextmanager(model, [(layer, batch_hook_base)]):
                    out_b = model(input_ids=input_ids_b, attention_mask=attn_mask_b)
                    logits_b = out_b.logits * coldness

                shift_logits_b = logits_b[:, :-1].contiguous()
                shift_labels_b = input_ids_b[:, 1:].contiguous()

                target_mask_b = torch.zeros_like(shift_labels_b, dtype=torch.bool)
                for row, sl in enumerate(steering_slices_b):
                    tgt_start_b = sl.start - 1
                    tgt_end_b = sl.stop - 1 if sl.stop is not None else shift_labels_b.size(1)
                    if tgt_start_b < tgt_end_b:
                        target_mask_b[row, tgt_start_b:tgt_end_b] = True

                active_logits_b = shift_logits_b[target_mask_b]
                active_labels_b = shift_labels_b[target_mask_b]
                if active_logits_b.numel() > 0:
                    log_probs_b = torch.log_softmax(active_logits_b, dim=-1)
                    idx = torch.arange(active_labels_b.size(0), device=active_labels_b.device)
                    p_b = torch.exp(log_probs_b[idx, active_labels_b])
                    base_loss = (-torch.log(torch.clamp(1.0 - p_b, min=eps))).mean() / math.log(10)
                else:
                    base_loss = torch.tensor(0.0, device=model.device)

                # cleanup base tensors
                del input_ids_b, attn_mask_b, shift_logits_b, shift_labels_b, active_logits_b, active_labels_b, out_b, logits_b
                torch.cuda.empty_cache()

            loss = ce_target + (base_loss_weight * base_loss if include_base_objective else 0.0)

            optim.zero_grad()
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params_to_opt, grad_clip)

            optim.step()

            # Advance the LR scheduler once per optimiser update
            if sched:
                sched.step()

            if max_norm is not None:
                if steering_type == "linear":
                    if vector.norm() > max_norm:
                        with torch.no_grad():
                            vector.mul_(max_norm / vector.norm())
                elif steering_type == "resid_lora":
                    # Enforce a joint norm constraint over (A, B, alpha)
                    with torch.no_grad():
                        A_norm = A.norm()
                        B_norm = B.norm()
                        a_norm = alpha_param.abs()
                        combined = torch.sqrt(A_norm * A_norm + B_norm * B_norm + a_norm * a_norm)
                        if combined > max_norm:
                            scale = max_norm / combined
                            A.mul_(scale)
                            B.mul_(scale)
                            alpha_param.mul_(scale)
                else:
                    raise ValueError(f"Unknown steering_type: {steering_type}")

            running_loss += loss.item()
            batches += 1
            loss_hist.append(loss.item())

            # ---- lightweight cleanup to mitigate CUDA OOM ----
            del input_ids, attn_mask, shift_logits, shift_labels, active_logits, active_labels, out, logits
            torch.cuda.empty_cache()

        batch_pbar.close()
        train_loss = running_loss / max(1, batches)

        # ---- Evaluation loss on eval set (if provided) ----
        eval_loss = None
        if have_eval:
            with torch.no_grad():
                running_eval_loss = 0.0
                eval_batches = 0
                for i in range(0, len(eval_prompts), optim_minibatch_size):
                    batch_indices = range(i, min(i + optim_minibatch_size, len(eval_prompts)))
                    
                    # Build right-padded eval batch
                    seqs_eval = [torch.cat([eval_prompt_tokens[j], eval_target_tokens[j]]) for j in batch_indices]
                    if not seqs_eval:
                        continue
                        
                    max_len_e = max(s.size(0) for s in seqs_eval)
                    pad_id = tokenizer.pad_token_id
                    input_ids_e = torch.full((len(seqs_eval), max_len_e), pad_id, device=model.device)
                    attn_mask_e = torch.zeros_like(input_ids_e)
                    steering_slices_e = []
                    
                    for row, j in enumerate(batch_indices):
                        L = seqs_eval[row].size(0)
                        input_ids_e[row, :L] = seqs_eval[row]
                        attn_mask_e[row, :L] = 1
                        start_e = eval_prompt_lens[j] if steering_token_window is None else (
                            eval_prompt_lens[j] + max(0, eval_target_lens[j] - steering_token_window)
                        )
                        steering_slices_e.append(slice(start_e, L))

                    def batch_hook_eval(_m, args, slices=steering_slices_e, stat_vecs=static_vectors_local):
                        (x,) = args
                        v_local = vector.to(x)
                        stat_vecs_on_device = [sv.to(x.device) for sv in stat_vecs]
                        for row, sl in enumerate(slices):
                            if projection_clamp:
                                seg = x[row, sl]
                                coef = (seg @ v_local) / (v_local.norm() ** 2)
                                x[row, sl] = seg - coef.unsqueeze(-1) * v_local + v_local
                            else:
                                x[row, sl] += v_local
                            if stat_vecs_on_device:
                                for sv in stat_vecs_on_device:
                                    x[row, sl] += sv
                        return (x,)

                    with hf_hooks_contextmanager(model, [(layer, batch_hook_eval)]):
                        out_e = model(input_ids=input_ids_e, attention_mask=attn_mask_e)
                        logits_e = out_e.logits * coldness

                    shift_logits_e = logits_e[:, :-1].contiguous()
                    shift_labels_e = input_ids_e[:, 1:].contiguous()

                    target_mask_e = torch.zeros_like(shift_labels_e, dtype=torch.bool)
                    for row, sl in enumerate(steering_slices_e):
                        tgt_start_e = sl.start - 1
                        tgt_end_e = sl.stop - 1 if sl.stop is not None else shift_labels_e.size(1)
                        if tgt_start_e < tgt_end_e:
                            target_mask_e[row, tgt_start_e:tgt_end_e] = True

                    active_logits_e = shift_logits_e[target_mask_e]
                    active_labels_e = shift_labels_e[target_mask_e]
                    ce_target_e = torch.nn.functional.cross_entropy(active_logits_e, active_labels_e, reduction="mean") / math.log(10)

                    base_loss_e = 0.0
                    if include_base_objective:
                        base_seqs_e = [torch.cat([eval_prompt_tokens[j], base_tokens_eval[j]]) for j in batch_indices]  # type: ignore
                        max_len_be = max(s.size(0) for s in base_seqs_e)
                        input_ids_be = torch.full((len(base_seqs_e), max_len_be), pad_id, device=model.device)
                        attn_mask_be = torch.zeros_like(input_ids_be)

                        steering_slices_be = []
                        for row, j in enumerate(batch_indices):
                            seq = base_seqs_e[row]
                            Lb = seq.size(0)
                            input_ids_be[row, :Lb] = seq
                            attn_mask_be[row, :Lb] = 1
                            start_be = eval_prompt_lens[j] if steering_token_window is None else (
                                eval_prompt_lens[j] + max(0, (base_lens_eval[j] if isinstance(base_lens_eval[j], int) else int(base_lens_eval[j])) - steering_token_window)  # type: ignore
                            )
                            steering_slices_be.append(slice(start_be, Lb))

                        def batch_hook_base_eval(_m, args, slices=steering_slices_be, stat_vecs=static_vectors_local):
                            (x,) = args
                            v_local = vector.to(x)
                            stat_vecs_on_device = [sv.to(x.device) for sv in stat_vecs]
                            for row, sl in enumerate(slices):
                                if projection_clamp:
                                    seg = x[row, sl]
                                    coef = (seg @ v_local) / (v_local.norm() ** 2)
                                    x[row, sl] = seg - coef.unsqueeze(-1) * v_local + v_local
                                else:
                                    x[row, sl] += v_local
                                if stat_vecs_on_device:
                                    for sv in stat_vecs_on_device:
                                        x[row, sl] += sv
                            return (x,)

                        with hf_hooks_contextmanager(model, [(layer, batch_hook_base_eval)]):
                            out_be = model(input_ids=input_ids_be, attention_mask=attn_mask_be)
                            logits_be = out_be.logits * coldness

                        shift_logits_be = logits_be[:, :-1].contiguous()
                        shift_labels_be = input_ids_be[:, 1:].contiguous()

                        target_mask_be = torch.zeros_like(shift_labels_be, dtype=torch.bool)
                        for row, sl in enumerate(steering_slices_be):
                            tgt_start_be = sl.start - 1
                            tgt_end_be = sl.stop - 1 if sl.stop is not None else shift_labels_be.size(1)
                            if tgt_start_be < tgt_end_be:
                                target_mask_be[row, tgt_start_be:tgt_end_be] = True

                        active_logits_be = shift_logits_be[target_mask_be]
                        active_labels_be = shift_labels_be[target_mask_be]
                        if active_logits_be.numel() > 0:
                            log_probs_be = torch.log_softmax(active_logits_be, dim=-1)
                            idx_be = torch.arange(active_labels_be.size(0), device=active_labels_be.device)
                            p_be = torch.exp(log_probs_be[idx_be, active_labels_be])
                            base_loss_e = (-torch.log(torch.clamp(1.0 - p_be, min=eps))).mean() / math.log(10)
                        else:
                            base_loss_e = torch.tensor(0.0, device=model.device)

                        # cleanup eval base tensors
                        del input_ids_be, attn_mask_be, shift_logits_be, shift_labels_be, active_logits_be, active_labels_be, out_be, logits_be
                        torch.cuda.empty_cache()

                    batch_eval_loss = (ce_target_e + (base_loss_weight * base_loss_e if include_base_objective else 0.0)).item()
                    running_eval_loss += batch_eval_loss
                    eval_batches += 1

                    # cleanup eval tensors
                    del input_ids_e, attn_mask_e, shift_logits_e, shift_labels_e, active_logits_e, active_labels_e, out_e, logits_e
                    torch.cuda.empty_cache()
                
                if eval_batches > 0:
                    eval_loss = running_eval_loss / eval_batches
                    eval_hist.append(eval_loss)

        # --- wandb logging ---
        if wandb_run is not None:
            log_payload = {
                'train_loss': train_loss,
                'step': step,
                'vector_norm': vector.norm().item(),
            }
            if eval_loss is not None:
                log_payload['eval_loss'] = eval_loss
            wandb_run.log(log_payload, step=step)

        metric = train_loss
        if metric + early_stopping_min_delta < best_loss:
            best_loss = metric
            if steering_type == "linear":
                best_vec = vector.detach().clone()
            else:
                best_lora = (A.detach().clone(), B.detach().clone(), alpha_param.detach().clone())
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                if debug:
                    print("Early stopping.")
                break

        # Scheduler is already stepped inside the minibatch loop

        # --- final cleanup for this step ---
        gc.collect()
        torch.cuda.empty_cache()

        current_lr = optim.param_groups[0]["lr"]
        pbar.set_description(
            f"step {step} lr {current_lr:.2e} train {train_loss:.4f}"
        )

    pbar.close()

    if steering_type == "linear":
        if best_vec is not None:
            vector.data.copy_(best_vec)
    else:
        if best_lora is not None:
            A.data.copy_(best_lora[0])
            B.data.copy_(best_lora[1])
            alpha_param.data.copy_(best_lora[2])

    if wandb_run is not None:
        if steering_type == "linear":
            wandb_run.log({'final_vector_norm': vector.norm().item()})
        else:
            wandb_run.log({'final_A_norm': A.norm().item(), 'final_B_norm': B.norm().item(), 'final_alpha': float(alpha_param.item())})

    if return_info:
        if steering_type == "linear":
            info = dict(
                iters=len(loss_hist),
                final_loss=best_loss,
                norm=vector.norm().item(),
                loss_history=loss_hist if return_loss_history else None,
                eval_loss_history=eval_hist if (return_loss_history and have_eval) else None,
            )
            return vector, info
        else:
            info = dict(
                iters=len(loss_hist),
                final_loss=best_loss,
                A_norm=A.norm().item(),
                B_norm=B.norm().item(),
                alpha=float(alpha_param.item()),
                loss_history=loss_hist if return_loss_history else None,
                eval_loss_history=eval_hist if (return_loss_history and have_eval) else None,
            )
            return {'A': A, 'B': B, 'alpha': alpha_param}, info
    return vector if steering_type == "linear" else {'A': A, 'B': B, 'alpha': alpha_param}


# Evaluation helper removed.
