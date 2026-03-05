# run_train.py
import os
import sys
from pathlib import Path

import yaml

from src.train import train_pipeline
from src.preprocess import set_global_determinism


def _load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg_path = os.environ.get("PFO_RS_CONFIG", "config.yaml")
    cfg = _load_config(cfg_path)

    # Reproducibility
    seed = int(cfg["project"].get("seed", 42))
    deterministic = bool(cfg["project"].get("deterministic", True))
    set_global_determinism(seed=seed, deterministic=deterministic)

    # Ensure output dirs exist
    out_root = Path(cfg["paths"]["outputs_dir"])
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    (out_root / "explanations").mkdir(parents=True, exist_ok=True)

    train_pipeline(cfg)

    print("\n✅ Training complete.")
    print(f"Model saved under: {cfg['paths']['models_dir']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
