from pathlib import Path
import pandas as pd
from src.config import DATASET_PATH


def load_neo_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Load the NASA NEO dataset from the datasets directory.

    Args:
        path (Path | None): Optional path to the dataset.
                           If None, uses DATASET_PATH from config.

    Returns:
        pd.DataFrame: Loaded NEO dataset.
    """
    dataset_path = path or DATASET_PATH
    return pd.read_csv(dataset_path)
