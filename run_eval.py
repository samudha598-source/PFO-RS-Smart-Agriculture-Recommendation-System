# run_eval.py
import os
import sys
from pathlib import Path
import yaml

from src.eval import evaluation_pipeline
from src.preprocess import set_global_determinism


def _load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg_path = os.environ.get("PFO_RS_CONFIG", "config.yaml")
    cfg = _load_config(cfg_path)

    seed = int(cfg["project"].get("seed", 42))
    deterministic = bool(cfg["project"].get("deterministic", True))
    set_global_determinism(seed=seed, deterministic=deterministic)

    # Ensure result directory exists
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)

    evaluation_pipeline(cfg)

    print("\n✅ Evaluation finished.")
    print(f"Results saved in: {cfg['paths']['results_dir']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
