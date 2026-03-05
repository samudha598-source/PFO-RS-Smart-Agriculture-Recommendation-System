# src/explain.py
# Explainability pipeline using SHAP and LIME for the PFO-RS system

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.data import load_dataset, split_dataset
from src.preprocess import preprocess_fit_transform, preprocess_transform
from src.features import select_features
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


def model_predict(model, device):
    """
    Wrapper used for SHAP and LIME to produce probability outputs.
    """

    def predict_fn(X):
        X = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    return predict_fn


def run_shap(model, device, X_train, X_sample, feature_names, output_dir, cfg):

    print("\nRunning SHAP analysis...")

    predict_fn = model_predict(model, device)

    background_size = min(cfg["explainability"]["shap"]["background_samples"], len(X_train))

    background = X_train[np.random.choice(len(X_train), background_size, replace=False)]

    explainer = shap.KernelExplainer(predict_fn, background)

    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
    )

    shap_path = output_dir / "shap_summary.png"

    plt.tight_layout()
    plt.savefig(shap_path)
    plt.close()

    print(f"SHAP plot saved to {shap_path}")


def run_lime(model, device, X_train, X_sample, feature_names, output_dir, cfg):

    print("\nRunning LIME explanations...")

    predict_fn = model_predict(model, device)

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=None,
        mode="classification",
    )

    lime_dir = output_dir / "lime_examples"
    lime_dir.mkdir(exist_ok=True)

    num_examples = min(cfg["explainability"]["lime"]["explain_samples"], len(X_sample))

    for i in range(num_examples):

        explanation = explainer.explain_instance(
            X_sample[i],
            predict_fn,
            num_features=cfg["explainability"]["lime"]["num_features"],
            num_samples=cfg["explainability"]["lime"]["num_samples"],
        )

        fig = explanation.as_pyplot_figure()

        fig_path = lime_dir / f"lime_{i}.png"

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    print(f"LIME explanations saved to {lime_dir}")


def explain_pipeline(cfg):

    print("\n===== PFO-RS Explainability Pipeline =====")

    device = get_device(cfg)

    # --------------------------
    # Load dataset
    # --------------------------
    df = load_dataset(cfg)

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_dataset(df, cfg)

    train_df = X_train_df.copy()
    train_df[cfg["data"]["target_col"]] = y_train

    test_df = X_test_df.copy()
    test_df[cfg["data"]["target_col"]] = y_test

    # --------------------------
    # Preprocessing
    # --------------------------
    X_train, y_train, preprocessor, feature_names = preprocess_fit_transform(cfg, train_df)
    X_test, y_test = preprocess_transform(cfg, test_df, preprocessor)

    # --------------------------
    # Load selected features
    # --------------------------
    feature_file = Path(cfg["paths"]["results_dir"]) / "selected_features.txt"

    if feature_file.exists():

        with open(feature_file) as f:
            selected_names = [line.strip() for line in f.readlines()]

        indices = [feature_names.index(f) for f in selected_names]

        X_train = select_features(X_train, indices)
        X_test = select_features(X_test, indices)

        feature_names = selected_names

    # --------------------------
    # Load model
    # --------------------------
    model_path = Path(cfg["paths"]["models_dir"]) / "pfo_rs_model.pt"

    num_classes = len(np.unique(y_train))

    model = load_model(
        model_path,
        X_train.shape[1],
        num_classes,
        cfg,
        device,
    )

    # --------------------------
    # Sampling for explanation
    # --------------------------
    sample_size = min(cfg["explainability"]["shap"]["explain_samples"], len(X_test))

    sample_idx = np.random.choice(len(X_test), sample_size, replace=False)

    X_sample = X_test[sample_idx]

    output_dir = Path(cfg["paths"]["explanations_dir"])

    # --------------------------
    # SHAP
    # --------------------------
    if cfg["explainability"]["shap"]["enabled"]:
        run_shap(model, device, X_train, X_sample, feature_names, output_dir, cfg)

    # --------------------------
    # LIME
    # --------------------------
    if cfg["explainability"]["lime"]["enabled"]:
        run_lime(model, device, X_train, X_sample, feature_names, output_dir, cfg)

    print("\nExplainability analysis completed.")
