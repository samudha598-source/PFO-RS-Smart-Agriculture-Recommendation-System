# src/eval.py
# Evaluation pipeline for the PFO-RS Smart Agriculture system

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.data import load_dataset, split_dataset
from src.preprocess import preprocess_fit_transform, preprocess_transform
from src.features import create_dataloader, select_features
from src.model import LSTMClassifier


def get_device(cfg):
    device_cfg = cfg["training"]["device"]

    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_cfg)


def load_model(model_path, input_dim, num_classes, cfg, device):
    model = LSTMClassifier(input_dim, num_classes, cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, loader, device):
    preds = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            logits = model(X)

            pred = torch.argmax(logits, dim=1).cpu().numpy()

            preds.extend(pred)
            targets.extend(y.numpy())

    return np.array(preds), np.array(targets)


def save_confusion_matrix(cm, labels, output_path):

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def evaluation_pipeline(cfg):

    print("\n===== PFO-RS Evaluation Pipeline =====")

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

    # --------------------------
    # Load selected features
    # --------------------------
    feature_file = Path(cfg["paths"]["results_dir"]) / "selected_features.txt"

    if feature_file.exists():

        with open(feature_file) as f:
            selected_names = [line.strip() for line in f.readlines()]

        indices = [feature_names.index(f) for f in selected_names]

        X_test = select_features(X_test, indices)

    # --------------------------
    # Load model
    # --------------------------
    model_path = Path(cfg["paths"]["models_dir"]) / "pfo_rs_model.pt"

    num_classes = len(np.unique(y_train))

    model = load_model(
        model_path,
        X_test.shape[1],
        num_classes,
        cfg,
        device,
    )

    # --------------------------
    # DataLoader
    # --------------------------
    test_loader = create_dataloader(X_test, y_test, cfg, shuffle=False)

    preds, targets = predict(model, test_loader, device)

    # --------------------------
    # Metrics
    # --------------------------
    acc = accuracy_score(targets, preds)

    precision = precision_score(
        targets,
        preds,
        average=cfg["evaluation"]["classification"]["average"],
        zero_division=0,
    )

    recall = recall_score(
        targets,
        preds,
        average=cfg["evaluation"]["classification"]["average"],
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        preds,
        average=cfg["evaluation"]["classification"]["average"],
        zero_division=0,
    )

    cm = confusion_matrix(targets, preds)

    print("\nEvaluation Metrics")
    print("-------------------")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # --------------------------
    # Save results
    # --------------------------
    results_dir = Path(cfg["paths"]["results_dir"])

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    metrics_path = results_dir / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    labels = np.unique(targets)

    cm_path = results_dir / "confusion_matrix.png"

    save_confusion_matrix(cm, labels, cm_path)

    print("\nResults saved.")
