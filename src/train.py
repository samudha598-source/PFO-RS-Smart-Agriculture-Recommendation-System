# src/train.py
# Training pipeline for the PFO-RS Smart Agriculture system

import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from src.data import load_dataset, split_dataset
from src.preprocess import preprocess_fit_transform, preprocess_transform
from src.features import create_dataloader, select_features, get_feature_subset
from src.foa import PermutationFlamingoOptimizer
from src.model import LSTMClassifier


def get_device(cfg):
    device_cfg = cfg["training"]["device"]

    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_cfg)


def build_model(input_dim, num_classes, cfg, device):
    model = LSTMClassifier(input_dim, num_classes, cfg)
    model.to(device)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).long()

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).long()

            logits = model(X)

            loss = criterion(logits, y)

            total_loss += loss.item()

    return total_loss / len(loader)


def compute_val_loss(X_train, y_train, X_val, y_val, cfg):
    """
    Fitness function used by FOA.
    Trains a small proxy model and returns validation loss.
    """

    device = torch.device("cpu")

    num_classes = len(np.unique(y_train))

    train_loader = create_dataloader(X_train, y_train, cfg)
    val_loader = create_dataloader(X_val, y_val, cfg, shuffle=False)

    model = LSTMClassifier(X_train.shape[1], num_classes, cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for _ in range(5):  # short proxy training
        train_epoch(model, train_loader, criterion, optimizer, device)

    val_loss = evaluate_epoch(model, val_loader, criterion, device)

    return val_loss


def train_pipeline(cfg):

    print("\n===== PFO-RS Training Pipeline =====")

    device = get_device(cfg)

    # --------------------------
    # Load dataset
    # --------------------------
    df = load_dataset(cfg)

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_dataset(df, cfg)

    train_df = X_train_df.copy()
    train_df[cfg["data"]["target_col"]] = y_train

    val_df = X_val_df.copy()
    val_df[cfg["data"]["target_col"]] = y_val

    test_df = X_test_df.copy()
    test_df[cfg["data"]["target_col"]] = y_test

    # --------------------------
    # Preprocessing
    # --------------------------
    X_train, y_train, preprocessor, feature_names = preprocess_fit_transform(cfg, train_df)
    X_val, y_val = preprocess_transform(cfg, val_df, preprocessor)
    X_test, y_test = preprocess_transform(cfg, test_df, preprocessor)

    print(f"\nTotal features after preprocessing: {len(feature_names)}")

    # --------------------------
    # Feature Selection (FOA)
    # --------------------------
    if cfg["feature_selection"]["enabled"]:

        print("\nRunning Permutation Flamingo Optimization...")

        foa = PermutationFlamingoOptimizer(
            num_features=X_train.shape[1],
            fitness_fn=lambda Xt, yt, Xv, yv: compute_val_loss(Xt, yt, Xv, yv, cfg),
            cfg=cfg,
        )

        selected_idx = foa.optimize(X_train, y_train, X_val, y_val)

        print(f"\nSelected {len(selected_idx)} features.")

        X_train = select_features(X_train, selected_idx)
        X_val = select_features(X_val, selected_idx)
        X_test = select_features(X_test, selected_idx)

        selected_features = get_feature_subset(feature_names, selected_idx)

    else:

        selected_features = feature_names

    # --------------------------
    # DataLoaders
    # --------------------------
    train_loader = create_dataloader(X_train, y_train, cfg)
    val_loader = create_dataloader(X_val, y_val, cfg, shuffle=False)

    num_classes = len(np.unique(y_train))

    model = build_model(X_train.shape[1], num_classes, cfg, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = None
    if cfg["training"]["scheduler"]["enabled"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg["training"]["scheduler"]["factor"],
            patience=cfg["training"]["scheduler"]["patience"],
        )

    # --------------------------
    # Training loop
    # --------------------------
    best_val = float("inf")
    patience_counter = 0

    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        val_loss = evaluate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0

            model_path = Path(cfg["paths"]["models_dir"]) / "pfo_rs_model.pt"
            torch.save(model.state_dict(), model_path)

        else:
            patience_counter += 1

        if cfg["training"]["early_stopping"]["enabled"]:
            if patience_counter >= cfg["training"]["early_stopping"]["patience"]:
                print("\nEarly stopping triggered.")
                break

    # --------------------------
    # Save artifacts
    # --------------------------
    results_dir = Path(cfg["paths"]["results_dir"])

    feature_file = results_dir / "selected_features.txt"

    with open(feature_file, "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    metrics_file = results_dir / "training_summary.json"

    summary = {
        "num_features_after_preprocessing": len(feature_names),
        "num_selected_features": len(selected_features),
        "best_validation_loss": float(best_val),
    }

    with open(metrics_file, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nTraining artifacts saved.")
