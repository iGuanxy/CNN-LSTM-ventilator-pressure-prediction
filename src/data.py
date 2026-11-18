"""Data loading, validation, stratified splitting, feature engineering and Dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, DATA_PATH, SEED, SAMPLE_NUM

CONT_FEATS = ["time_step", "u_in"]
CAT_FEATS = ["R", "C"]
TARGET_COL = "pressure"


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and sort by breath_id and time_step."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path.resolve()}\n"
            "Download train.csv with kagglehub into the data/ directory, "
            "or set the VENTILATOR_DATA_PATH environment variable."
        )
    df = pd.read_csv(path)
    return df.sort_values(by=["breath_id", "time_step"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Check missing values, duplicate rows, and whether every cycle has 80 steps."""
    print("Raw data shape:", df.shape)
    print("Missing values:\n", df.isnull().sum().to_string())
    print("Duplicate rows:", df.duplicated().sum())
    breath_len = df.groupby("breath_id").size()
    print("Cycle length distribution:", breath_len.value_counts().sort_index().to_dict())
    assert (breath_len == 80).all(), "Found breathing cycles whose length is not 80"


def build_datasets(df: pd.DataFrame, sample_num: int = SAMPLE_NUM, seed: int = SEED):
    """Stratified sampling by R/C and train/val/local-test split.

    Returns (datasets, number of feature columns).
    """
    breath_rc = df[["breath_id", "R", "C"]].drop_duplicates().reset_index(drop=True)
    sample_num = min(sample_num, len(breath_rc))
    breath_sample = breath_rc.sample(n=sample_num, random_state=seed).reset_index(drop=True)
    selected = breath_sample["breath_id"].values
    df_sub = df[df["breath_id"].isin(selected)].copy()

    all_bids = df_sub["breath_id"].unique()
    strata = dict(zip(breath_sample["breath_id"],
                      breath_sample["R"].astype(str) + "_" + breath_sample["C"].astype(str)))
    bid_train, bid_temp = train_test_split(
        all_bids, test_size=0.3, random_state=seed, shuffle=True,
        stratify=[strata[b] for b in all_bids],
    )
    bid_val, bid_test = train_test_split(
        bid_temp, test_size=0.5, random_state=seed, shuffle=True,
        stratify=[strata[b] for b in bid_temp],
    )
    print(f"Sampled cycles: {len(selected)} | train {len(bid_train)} / "
          f"val {len(bid_val)} / test {len(bid_test)}")

    parts = {
        "train": df_sub[df_sub["breath_id"].isin(bid_train)].copy(),
        "val": df_sub[df_sub["breath_id"].isin(bid_val)].copy(),
        "test": df_sub[df_sub["breath_id"].isin(bid_test)].copy(),
    }

    # Fit scaler and encoder on the training set only
    scaler = StandardScaler()
    parts["train"][CONT_FEATS] = scaler.fit_transform(parts["train"][CONT_FEATS])
    for key in ("val", "test"):
        parts[key][CONT_FEATS] = scaler.transform(parts[key][CONT_FEATS])

    ohe = OneHotEncoder(sparse_output=False, drop=None)
    ohe.fit(parts["train"][CAT_FEATS])
    for key in parts:
        rc = ohe.transform(parts[key][CAT_FEATS])
        rc_cols = [f"rc_{i}" for i in range(rc.shape[1])]
        parts[key][rc_cols] = rc

    feat_cols = CONT_FEATS + ["u_out"] + [f"rc_{i}" for i in range(ohe.transform(parts["train"][CAT_FEATS]).shape[1])]
    datasets = {
        key: VentDataset(part, feat_cols, TARGET_COL) for key, part in parts.items()
    }
    return datasets, len(feat_cols)


class VentDataset(Dataset):
    """Each sample is one complete 80-step breathing cycle."""

    def __init__(self, data_df: pd.DataFrame, feat_cols: list[str], target_col: str):
        if not np.isin(np.unique(data_df["u_out"].values), [0, 1]).all():
            raise ValueError("u_out must be 0/1, otherwise the inhalation mask is invalid")
        self.x_list, self.y_list, self.mask_list = [], [], []
        for _, group in data_df.sort_values(["breath_id", "time_step"]).groupby("breath_id"):
            self.x_list.append(group[feat_cols].values.astype(np.float32))
            self.y_list.append(group[target_col].values.astype(np.float32))
            self.mask_list.append(group["u_out"].values.astype(np.float32))

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_list[idx]),
            torch.tensor(self.y_list[idx]),
            torch.tensor(self.mask_list[idx]),
        )


def make_loaders(datasets, batch_size: int = BATCH_SIZE):
    """Build DataLoaders for train / val / test."""
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            drop_last=len(datasets["train"]) >= batch_size),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
