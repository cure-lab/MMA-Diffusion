# mma_runner.py - SD3.5 version
import argparse
import os
import random
from typing import Any, List

from datasets import load_dataset
from text_space_attack.MMA_para import ParallelMMA
from text_space_attack.model import GenerativeModel, get_model_path

try:
    import swanlab
except ImportError:
    swanlab = None


DEFAULT_DATASET = "YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark"
ATTACK_KWARGS = {
    "batch_size",
    "topk",
    "n_steps",
    "control_length",
    "threshold",
    "early_stopping",
    "lambda_vec",
    "lambda_ctxt",
    "use_context_loss",
}


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_runner(args: argparse.Namespace, use_safe_checker: bool, model_name: str) -> ParallelMMA:
    """Create the target generative model and the ParallelMMA attacker."""
    model_path = get_model_path(model_name)
    model = GenerativeModel(model_path=model_path, use_safe_checker=use_safe_checker)

    subdir = "with_safe_checker" if use_safe_checker else "no_safe_checker"
    save_dir = os.path.join(model_name, subdir)
    os.makedirs(save_dir, exist_ok=True)

    attack_kwargs = {key: getattr(args, key) for key in ATTACK_KWARGS}
    return ParallelMMA(
        target_model=model,
        text_detector=None,
        image_detector=None,
        save_dir=save_dir,
        **attack_kwargs,
    )


def load_inputs(args: argparse.Namespace) -> List[str]:
    """Load target prompts from a local CSV or the public benchmark."""
    if args.data_file:
        ds = load_dataset("csv", data_files=args.data_file, split=args.dataset_split)
    else:
        ds = load_dataset(args.dataset_name, split=args.dataset_split)

    if args.prompt_column not in ds.column_names:
        raise ValueError(
            f"Prompt column '{args.prompt_column}' not found. "
            f"Available columns: {', '.join(ds.column_names)}"
        )

    return ds[args.prompt_column][: args.limit]


def run_once(
    args: argparse.Namespace,
    use_safe_checker: bool,
    model_name: str = "stable-diffusion-3.5-medium",
) -> Any:
    """Execute one batch attack."""
    runner = build_runner(args, use_safe_checker=use_safe_checker, model_name=model_name)
    inputs = load_inputs(args)
    return runner.attack(inputs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MMA attack on SD3.5.")
    p.add_argument("--use-safe-checker", type=str_to_bool, default=False)
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--model_name", type=str, default="stable-diffusion-3.5-medium")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--n_steps", type=int, default=500)
    p.add_argument("--control_length", type=int, default=20)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--early_stopping", type=str_to_bool, default=False)
    p.add_argument("--lambda_vec", type=float, default=1.0, help="Weight for pooled vector loss.")
    p.add_argument("--lambda_ctxt", type=float, default=0.5, help="Weight for context sequence loss.")
    p.add_argument("--use_context_loss", type=str_to_bool, default=True)
    p.add_argument("--seed", type=int, default=42, help="Random seed for prompt initialization.")
    p.add_argument("--data_file", type=str, default=None, help="Optional local CSV file.")
    p.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--prompt_column", type=str, default="target_prompt")
    p.add_argument("--disable_swanlab", action="store_true")
    p.add_argument("--swanlab_project", type=str, default="MMA_SD3.5")
    p.add_argument("--swanlab_workspace", type=str, default=None)
    return p.parse_args()


def init_swanlab(args: argparse.Namespace) -> None:
    if args.disable_swanlab or swanlab is None:
        return

    init_kwargs = {"project": args.swanlab_project}
    if args.swanlab_workspace:
        init_kwargs["workspace"] = args.swanlab_workspace
    swanlab.init(**init_kwargs)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    init_swanlab(args)

    print(f"[SD3.5 MMA Attack] Args: {args}")
    results = run_once(
        args,
        use_safe_checker=args.use_safe_checker,
        model_name=args.model_name,
    )
    if hasattr(results, "__len__"):
        print(f"[INFO] Completed. Results length: {len(results)}")
    else:
        print("[INFO] Completed.")
