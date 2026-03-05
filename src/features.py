# src/features.py
# Feature preparation utilities and PyTorch dataset helpers

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    """
    PyTorch Dataset wrapper for tabular data.
    Supports both classification and optional regression targets.
    """

    def __init__(self, X, y, sequence_mode=False, seq_len=1):
        """
        Parameters
        ----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        sequence_mode : bool
            Whether to reshape features into sequence format
        seq_len : int
            Length of sequence window
        """

        self.sequence_mode = sequence_mode
        self.seq_len = seq_len

        if sequence_mode:
            self.X = self._create_sequences(X, seq_len)
            self.y = y[seq_len - 1 :]
        else:
            self.X = X
            self.y = y

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y)

    def _create_sequences(self, X, seq_len):
        """
        Converts tabular features into sliding window sequences.
        """
        sequences = []

        for i in range(len(X) - seq_len + 1):
            window = X[i : i + seq_len]
            sequences.append(window)

        return np.array(sequences)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloader(X, y, cfg, shuffle=True):
    """
    Creates a PyTorch DataLoader.
    """

    sequence_enabled = cfg["model"]["sequence"]["enabled"]
    seq_len = cfg["model"]["sequence"]["seq_len"]

    dataset = TabularDataset(
        X,
        y,
        sequence_mode=sequence_enabled,
        seq_len=seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["training"]["num_workers"],
        drop_last=False,
    )

    return loader


def select_features(X, feature_indices):
    """
    Select subset of features after FOA optimization.
    """

    return X[:, feature_indices]


def get_feature_subset(feature_names, indices):
    """
    Returns selected feature names for reporting.
    """

    return [feature_names[i] for i in indices]
