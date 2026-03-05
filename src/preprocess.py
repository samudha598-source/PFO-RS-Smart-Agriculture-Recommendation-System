# src/preprocess.py
# Data preprocessing utilities and reproducibility helpers

import random
import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def set_global_determinism(seed=42, deterministic=True):
    """
    Ensures reproducible experiments across numpy, torch, and python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def identify_column_types(df):
    """
    Detect numeric and categorical columns automatically.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_cols, categorical_cols


def build_preprocessor(cfg, df):
    """
    Builds sklearn preprocessing pipeline based on config.yaml.
    """
    target_col = cfg["data"]["target_col"]
    drop_cols = cfg["data"]["drop_cols"]

    df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[target_col])

    numeric_cols, categorical_cols = identify_column_types(X)

    # --- Missing value strategies ---
    num_strategy = cfg["data"]["missing"]["numeric_strategy"]
    cat_strategy = cfg["data"]["missing"]["categorical_strategy"]

    # --- Scaling strategy ---
    scaler_type = cfg["data"]["scaling"]["numeric_scaler"]

    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    # --- Numeric pipeline ---
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=num_strategy)),
            ("scaler", scaler),
        ]
    )

    # --- Categorical pipeline ---
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cat_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def preprocess_fit_transform(cfg, df):
    """
    Fit preprocessing pipeline on training data and transform.
    """
    target_col = cfg["data"]["target_col"]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(cfg, df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_processed = preprocessor.fit_transform(X)

    feature_names = numeric_cols

    if len(categorical_cols) > 0:
        encoder = preprocessor.named_transformers_["cat"]["encoder"]
        cat_features = encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names = numeric_cols + cat_features

    return X_processed, y.values, preprocessor, feature_names


def preprocess_transform(cfg, df, preprocessor):
    """
    Transform dataset using fitted preprocessor.
    """
    target_col = cfg["data"]["target_col"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_processed = preprocessor.transform(X)

    return X_processed, y.values
