
import argparse
import torch
import gc
import sys
import os
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import utils

def main():
    """
    Main function to generate and cache model activations for specified layers.
    """
    parser = argparse.ArgumentParser(description="Generate and cache model activations.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use for generating activations (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B').")
    parser.add_argument("--layers", type=int, nargs='+', required=True,
                        help="A list of one or more layer numbers to process.")
    parser.add_argument("--n_examples", type=int, default=500,
                        help="Number of examples to use for generating activations.")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load the model in 8-bit mode to save memory.")

    args = parser.parse_args()

    print(f"Generating activations for model: {args.model}")
    print(f"Processing layers: {args.layers}")
    print(f"Number of examples: {args.n_examples}")

    # Load the model and tokenizer
    try:
        model, tokenizer = utils.load_model(
            model_name=args.model,
            load_in_8bit=args.load_in_8bit
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process saved responses to generate and cache activations with strict validation
    try:
        results = utils.process_saved_responses(
            model_name=args.model,
            n_examples=args.n_examples,
            model=model,
            tokenizer=tokenizer,
            layer_or_layers=args.layers
        )

        # Normalize return structure to {layer: (activations, texts)}
        if isinstance(results, tuple):
            assert len(args.layers) == 1, "Expected a single layer when results is a tuple"
            results_by_layer = {int(args.layers[0]): results}
        else:
            results_by_layer = results

        # Validate each layer's activations strictly
        for layer, (activations, texts) in results_by_layer.items():
            assert isinstance(activations, np.ndarray), f"Layer {layer}: activations must be a numpy array"
            assert activations.ndim == 2, f"Layer {layer}: expected 2D activations, got {activations.ndim}D"
            nonfinite = int(np.sum(~np.isfinite(activations)))
            min_val = float(np.nanmin(activations))
            max_val = float(np.nanmax(activations))
            mean_val = float(np.nanmean(activations))
            print(f"Layer {layer}: activations shape={activations.shape}, nonfinite={nonfinite}, min/mean/max={min_val:.6f}/{mean_val:.6f}/{max_val:.6f}")
            assert nonfinite == 0, f"Layer {layer}: found {nonfinite} non-finite activation values"
            row_norms = np.linalg.norm(activations, axis=1)
            assert np.isfinite(row_norms).all(), f"Layer {layer}: non-finite row norms detected"
            min_norm = float(np.min(row_norms))
            max_norm = float(np.max(row_norms))
            mean_norm = float(np.mean(row_norms))
            print(f"Layer {layer}: row-norms min/mean/max={min_norm:.6f}/{mean_norm:.6f}/{max_norm:.6f}")
            # Expect near-unit norms after normalization; fail fast if clearly broken
            assert min_norm > 0.0, f"Layer {layer}: zero row norm detected"
            assert max_norm < 10.0, f"Layer {layer}: excessively large row norm {max_norm:.6f} indicates normalization failure"

        print("Successfully generated and cached activations with valid numerics.")
    except Exception as e:
        print(f"Error processing saved responses: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleaned up resources.")

if __name__ == "__main__":
    main() 