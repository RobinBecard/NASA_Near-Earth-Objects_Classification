from pathlib import Path
import pandas as pd
from src.config import DATASET_PATH


def load_neo_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the NASA NEO dataset from the datasets directory."""
    dataset_path = path or DATASET_PATH
    return pd.read_csv(dataset_path)
