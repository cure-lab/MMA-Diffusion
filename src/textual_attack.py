# mma_runner.py
import os
import argparse
from typing import List, Any

from datasets import load_dataset
from text_space_attack.model import GenerativeModel, get_model_path
from text_space_attack.MMA_para import ParallelMMA


def build_runner(args: argparse.Namespace, use_safe_checker: bool, model_name: str) -> ParallelMMA:
    """
    Create the target generative model and the ParallelMMA attacker, then return the runner.

    - use_safe_checker: whether to enable the model's internal safety checker (passed to GenerativeModel)
    - model_name: the textual identifier of the target model
    """
    # Build the generative model (safety checker is controlled by the switch below)
    model_path = get_model_path(model_name)
    model = GenerativeModel(model_path=model_path, use_safe_checker=use_safe_checker)

    # Use a separate save directory for each safety-checker setting
    subdir = "with_safe_checker" if use_safe_checker else "no_safe_checker"
    save_dir = os.path.join(model_name, subdir)
    os.makedirs(save_dir, exist_ok=True)

    # You can plug in real detectors here; keep None for now
    runner = ParallelMMA(
        target_model=model,
        text_detector=None,
        image_detector=None,
        save_dir=save_dir,
        **vars(args),  # unpack CLI args into kwargs expected by ParallelMMA
    )
    return runner


def load_inputs(limit: int) -> List[str]:
    """
    Load target prompts from the public benchmark and return the first `limit` entries.
    Requires internet access for Hugging Face datasets.
    """
    ds = load_dataset(
        "YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark",
        split="train",
    )
    prompts = ds["target_prompt"][:limit]
    return prompts


def run_once(
    args: argparse.Namespace,
    use_safe_checker: bool,
    limit: int = 50,
    model_name: str = "stable-diffusion-1_5",
) -> Any:
    """
    Execute one batch attack:
      1) Build the runner (GenerativeModel + ParallelMMA)
      2) Load the first `limit` target prompts
      3) Run the attack
    Returns the result from `ParallelMMA.attack(...)`.
    """
    runner = build_runner(args, use_safe_checker=use_safe_checker, model_name=model_name)
    inputs = load_inputs(limit=limit)
    results = runner.attack(inputs)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MMA with/without a safety checker.")
    p.add_argument(
        "--use-safe-checker",
        type=str,
        default="true",
        choices=["false", "true"],
        help="Whether to enable the GenerativeModel's internal safety checker.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="How many prompts to run.",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="stable-diffusion-v1-5",
        help="Target model name.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the attacker.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=256,
        help="Number of candidate tokens to consider.",
    )
    p.add_argument(
        "--n_steps",
        type=int,
        default=1000,
        help="Number of optimization steps.",
    )
    p.add_argument(
        "--control_length",
        type=int,
        default=20,
        help="Number of control tokens.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for early decisions.",
    )
    p.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping (if supported by the attacker).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_safe_checker = (args.use_safe_checker.lower() == "true")
    print(args)
    results = run_once(
        args,
        use_safe_checker=use_safe_checker,
        limit=args.limit,
        model_name=args.model_name,
    )
    # Avoid dumping the entire result blob; print a concise summary instead.
    if hasattr(results, "__len__"):
        print(f"[INFO] Completed. Results length: {len(results)}")
    else:
        print("[INFO] Completed.")
