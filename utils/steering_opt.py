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
    warmup_steps: int = 0,
    min_lr: float = 0.0,
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
):
    """One‑stop promotion‑steering optimiser matching the paper’s details."""

    # ---------------- Pre‑flight checks ---------------- #
    if len(prompts) != len(target_completions):
        raise ValueError("Prompts and completions length mismatch.")
    # Evaluation logic removed; optimisation now relies only on training loss.

    # ---------------- Vector initialisation ------------- #
    d_model = model.config.hidden_size
    vector = torch.randn(d_model, device=model.device)
    vector = starting_norm * vector / vector.norm()
    vector.requires_grad_(True)

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

    # Evaluation dataset and metrics have been removed.
    eval_hist = []

    # ---------------- Optimiser & LR schedule ----------- #
    optim = torch.optim.Adam([vector], lr=lr)

    # Ensure static_vectors is a list of tensors on correct device
    if static_vectors is None:
        static_vectors_local: list[torch.Tensor] = []
    else:
        static_vectors_local = [sv.to(model.device).detach() for sv in static_vectors]

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
    best_vec, best_loss = None, float("inf")
    loss_hist = []
    patience = 0

    pbar = tqdm(range(max_iters), desc="")

    for step in pbar:
        # ----- minibatch shuffling -----
        idxs = list(range(len(prompts)))
        random.shuffle(idxs)

        running_loss, batches = 0.0, 0

        for bs in range(0, len(idxs), optim_minibatch_size):
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

                    # Add static vectors (always additive, no projection-clamp)
                    if stat_vecs_on_device:
                        for sv in stat_vecs_on_device:
                            x[row, sl] += sv
                return (x,)

            with hf_hooks_contextmanager(model, [(layer, batch_hook)]):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits * 0.7  # coldness

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
            loss = torch.nn.functional.cross_entropy(active_logits, active_labels, reduction="mean") / math.log(10)  # hartleys

            optim.zero_grad()
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_([vector], grad_clip)

            optim.step()

            # Advance the LR scheduler once per optimiser update
            if sched:
                sched.step()

            if max_norm is not None and vector.norm() > max_norm:
                with torch.no_grad():
                    vector.mul_(max_norm / vector.norm())

            running_loss += loss.item()
            batches += 1
            loss_hist.append(loss.item())

            # ---- lightweight cleanup to mitigate CUDA OOM ----
            del input_ids, attn_mask, shift_logits, shift_labels, active_logits, active_labels, out, logits
            torch.cuda.empty_cache()

        train_loss = running_loss / max(1, batches)

        # Evaluation step removed; rely solely on train_loss.
        eval_loss = None

        # --- wandb logging ---
        if wandb_run is not None:
            wandb_run.log({
                'train_loss': train_loss,
                'step': step,
                'vector_norm': vector.norm().item(),
            }, step=step)

        metric = train_loss
        if metric + early_stopping_min_delta < best_loss:
            best_loss = metric
            best_vec = vector.detach().clone()
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

    if best_vec is not None:
        vector.data.copy_(best_vec)

    if wandb_run is not None:
        wandb_run.log({'final_vector_norm': vector.norm().item()})

    if return_info:
        info = dict(
            iters=len(loss_hist),
            final_loss=best_loss,
            norm=vector.norm().item(),
            loss_history=loss_hist if return_loss_history else None,
        )
        return vector, info
    return vector


# Evaluation helper removed.
