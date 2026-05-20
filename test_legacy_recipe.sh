#!/usr/bin/env bash
# ============================================================
# End-to-end test of the legacy CE recipe with TINY data budget.
# Tests:
#   1. Data loading (annotated responses, responses)
#   2. Bias vector training (2 iters, 5 examples)
#   3. Cat vector training (2 iters, 5 examples) - idx 0 only
#   4. layer_map.json creation
#   5. Vector sanity checks (file exists, loadable, correct keys)
#   6. hybrid_eval.py vector loading in dry-run mode (--n_tasks 2)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash test_legacy_recipe.sh
# ============================================================
set -e

GPU=${CUDA_VISIBLE_DEVICES:-0}
echo "Using GPU: $GPU"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/train-vectors"
HYBRID_DIR="$SCRIPT_DIR/hybrid"

# Use a separate save dir to avoid polluting production vectors
TEST_SAVE_DIR="$TRAIN_DIR/results/vars/optimized_vectors_test"
rm -rf "$TEST_SAVE_DIR"
mkdir -p "$TEST_SAVE_DIR"

echo ""
echo "========================================"
echo " STEP 0: Verify input data files"
echo "========================================"

# Check responses files
for f in \
    "generate-responses/results/vars/responses_open-reasoner-zero-1.5b.json" \
    "generate-responses/results/vars/annotated_responses_open-reasoner-zero-1.5b.json" \
    "generate-responses/results/vars/responses_open-reasoner-zero-0.5b.json" \
    "generate-responses/results/vars/annotated_responses_open-reasoner-zero-0.5b.json"; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        echo "  OK: $f"
    else
        echo "  ERROR: Missing $f"
        exit 1
    fi
done

# Check SAE files
for f in \
    "train-saes/results/vars/saes/sae_open-reasoner-zero-1.5b_layer8_clusters5.pt" \
    "train-saes/results/vars/saes/sae_open-reasoner-zero-0.5b_layer8_clusters10.pt"; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
        echo "  OK: $f"
    else
        echo "  ERROR: Missing $f"
        exit 1
    fi
done
echo "All input data verified ✓"

echo ""
echo "========================================"
echo " STEP 1: Test bias training (ORZ-1.5B, 2 iters)"
echo "========================================"

cd "$TRAIN_DIR"

N_TRAIN=5 N_EVAL=2 MAX_ITERS=2 \
CUDA_VISIBLE_DEVICES="$GPU" python optimize_steering_vectors.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --max_iters 2 \
    --n_training_examples 5 \
    --n_eval_examples 2 \
    --optim_minibatch_size 2 \
    --layer 10 \
    --steering_vector_idx -1 \
    --lr "1e-2" \
    --save_path "$TEST_SAVE_DIR"

echo ""
if [[ -f "$TEST_SAVE_DIR/qwen2.5-1.5b_bias_linear.pt" ]]; then
    echo "  PASS: Bias vector file created ✓"
else
    echo "  FAIL: Bias vector file NOT created"
    exit 1
fi

echo ""
echo "========================================"
echo " STEP 2: Test cat idx0 training (ORZ-1.5B, 2 iters)"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPU" python optimize_steering_vectors.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --max_iters 2 \
    --n_training_examples 5 \
    --n_eval_examples 2 \
    --optim_minibatch_size 2 \
    --layer 10 \
    --steering_vector_idx 0 \
    --lr "1e-2" \
    --use_activation_perplexity_selection \
    --save_path "$TEST_SAVE_DIR"

echo ""
if [[ -f "$TEST_SAVE_DIR/qwen2.5-1.5b_idx0_linear.pt" ]]; then
    echo "  PASS: Cat idx0 file created ✓"
else
    echo "  FAIL: Cat idx0 file NOT created"
    exit 1
fi

echo ""
echo "========================================"
echo " STEP 3: Test write_layer_map.py"
echo "========================================"

python write_layer_map.py --save_dir "$TEST_SAVE_DIR" --layer 10 --n_cats 5

if [[ -f "$TEST_SAVE_DIR/layer_map.json" ]]; then
    echo "  PASS: layer_map.json created ✓"
    cat "$TEST_SAVE_DIR/layer_map.json"
else
    echo "  FAIL: layer_map.json NOT created"
    exit 1
fi

echo ""
echo "========================================"
echo " STEP 4: Vector sanity check (Python)"
echo "========================================"

cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" python -c "
import torch, json, os, sys

save_dir = 'train-vectors/results/vars/optimized_vectors_test'

# Check bias
bias_path = os.path.join(save_dir, 'qwen2.5-1.5b_bias_linear.pt')
bias_obj = torch.load(bias_path, map_location='cpu', weights_only=False)
assert isinstance(bias_obj, dict), 'bias should be a dict'
assert 'bias' in bias_obj, f'bias dict keys: {list(bias_obj.keys())}'
bias_vec = bias_obj['bias']
assert bias_vec.ndim == 1, f'bias should be 1D, got shape {bias_vec.shape}'
print(f'  bias: shape={tuple(bias_vec.shape)}, norm={bias_vec.norm().item():.4f}')

# Check cat
cat0_path = os.path.join(save_dir, 'qwen2.5-1.5b_idx0_linear.pt')
cat0_obj = torch.load(cat0_path, map_location='cpu', weights_only=False)
assert isinstance(cat0_obj, dict), 'cat0 should be a dict'
assert 'idx0' in cat0_obj, f'cat0 dict keys: {list(cat0_obj.keys())}'
cat0_vec = cat0_obj['idx0']
assert cat0_vec.ndim == 1, f'cat0 should be 1D, got shape {cat0_vec.shape}'
assert cat0_vec.shape == bias_vec.shape, 'cat0 and bias should match shape'
print(f'  idx0: shape={tuple(cat0_vec.shape)}, norm={cat0_vec.norm().item():.4f}')

# Check layer_map
lm = json.load(open(os.path.join(save_dir, 'layer_map.json')))
assert lm == {'idx0': 10, 'idx1': 10, 'idx2': 10, 'idx3': 10, 'idx4': 10}, f'bad layer_map: {lm}'
print(f'  layer_map: {lm}')

print('')
print('PASS: Vector sanity checks ✓')
"
echo "  Python sanity checks done ✓"

echo ""
echo "========================================"
echo " STEP 5: Fake remaining 4 cats (copy idx0 -> idx1..idx4)"
echo "========================================"
# So hybrid_eval.py can load all 5 cats without needing full training

cd "$SCRIPT_DIR"
python -c "
import torch, os

save_dir = 'train-vectors/results/vars/optimized_vectors_test'
src = torch.load(os.path.join(save_dir, 'qwen2.5-1.5b_idx0_linear.pt'), map_location='cpu', weights_only=False)

for i in range(1, 5):
    fake = {'idx' + str(i): src['idx0'].clone()}
    torch.save(fake, os.path.join(save_dir, f'qwen2.5-1.5b_idx{i}_linear.pt'))
    print(f'  Created fake idx{i} from idx0')
print('Fake cats created (for eval load test only)')
"

echo ""
echo "========================================"
echo " STEP 6: Dry-run hybrid_eval.py (n_tasks=2)"
echo "========================================"

cd "$HYBRID_DIR"

CUDA_VISIBLE_DEVICES="$GPU" python hybrid_eval.py \
    --base_model "Qwen/Qwen2.5-1.5B" \
    --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B" \
    --dataset math500 \
    --sae_layer 8 \
    --n_clusters 5 \
    --n_tasks 2 \
    --batch_gen_size 2 \
    --hybrid_gen_batch_size 2 \
    --dom_vectors_dir "../train-vectors/results/vars/optimized_vectors_test" \
    --old_vectors_dir "../train-vectors/results/vars/optimized_vectors_test" \
    --old_vectors_layer 10 \
    --bias_vector_path "../train-vectors/results/vars/optimized_vectors_test/qwen2.5-1.5b_bias_linear.pt" \
    --fixed_coef 1.0 \
    --coef_select fixed \
    --steer_all_positions_full \
    --results_suffix "test_run_sanity" \
    --no_response_cache

echo ""
echo "========================================"
echo " ALL TESTS PASSED ✓"
echo "========================================"
echo ""
echo "Cleaning up test vectors..."
rm -rf "$TEST_SAVE_DIR"
echo "Done. Ready to run full training."
