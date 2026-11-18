"""Global configuration: paths, random seed, hyperparameters."""
import os
from pathlib import Path

# Project root directory (parent of this file)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data path: environment variable takes precedence, otherwise data/train.csv
DATA_PATH = Path(
    os.environ.get("VENTILATOR_DATA_PATH", ROOT_DIR / "data" / "train.csv")
)

FIG_DIR = ROOT_DIR / "figures"
WEIGHT_DIR = ROOT_DIR / "weights"
FIG_DIR.mkdir(exist_ok=True)
WEIGHT_DIR.mkdir(exist_ok=True)

SEED = 42
SAMPLE_NUM = 8000          # number of breathing cycles sampled from the full dataset
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
HIDDEN_DIM = 64
INPUT_FEAT_DIM = 9

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
