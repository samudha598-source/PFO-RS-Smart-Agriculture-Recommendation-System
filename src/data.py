# src/data.py
# Dataset loading and reproducible train/val/test splitting

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(cfg):
    """
    Load dataset from CSV defined in config.yaml.
    """
    dataset_path = Path(cfg["paths"]["dataset_csv"])

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please place the dataset in the configured path."
        )

    df = pd.read_csv(dataset_path)

    if cfg["logging"]["verbose"]:
        print(f"\nLoaded dataset from {dataset_path}")
        print(f"Dataset shape: {df.shape}")

    return df


def split_dataset(df, cfg):
    """
    Perform train/validation/test split with optional stratification.
    """
    target_col = cfg["data"]["target_col"]
    split_cfg = cfg["data"]["split"]

    test_size = split_cfg["test_size"]
    val_size = split_cfg["val_size"]
    stratify_enabled = split_cfg["stratify"]
    shuffle = split_cfg["shuffle"]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if stratify_enabled else None

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=cfg["project"]["seed"],
        stratify=stratify,
        shuffle=shuffle,
    )

    # Validation split from training set
    val_relative_size = val_size / (1 - test_size)

    stratify_val = y_train if stratify_enabled else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_relative_size,
        random_state=cfg["project"]["seed"],
        stratify=stratify_val,
        shuffle=shuffle,
    )

    if cfg["logging"]["verbose"]:
        print("\nDataset split summary:")
        print(f"Train size: {len(X_train)}")
        print(f"Validation size: {len(X_val)}")
        print(f"Test size: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_split(cfg):
    """
    Convenience wrapper used by training and evaluation pipelines.
    """
    df = load_dataset(cfg)
    return split_dataset(df, cfg)
