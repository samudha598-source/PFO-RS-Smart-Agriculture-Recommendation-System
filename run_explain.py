# run_explain.py
import os
import sys
from pathlib import Path
import yaml

from src.explain import explain_pipeline
from src.preprocess import set_global_determinism


def _load_config(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg_path = os.environ.get("PFO_RS_CONFIG", "config.yaml")
    cfg = _load_config(cfg_path)

    # Set reproducibility
    seed = int(cfg["project"].get("seed", 42))
    deterministic = bool(cfg["project"].get("deterministic", True))
    set_global_determinism(seed=seed, deterministic=deterministic)

    # Ensure explanation output directory exists
    Path(cfg["paths"]["explanations_dir"]).mkdir(parents=True, exist_ok=True)

    explain_pipeline(cfg)

    print("\n✅ Explainability analysis completed.")
    print(f"Outputs saved in: {cfg['paths']['explanations_dir']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
