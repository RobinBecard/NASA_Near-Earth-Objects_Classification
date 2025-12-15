from pathlib import Path
import pandas as pd
from src.config import get_config


class NEODataLoader:
    """
    Class responsible for loading the NASA NEO dataset.
    """

    def __init__(self, path: Path | None = None):
        """
        Initialize the NEO Data Loader.

        Args:
            path (Path | None): Optional path to the dataset. If None, uses path from config.yaml.
        """
        if path is None:
            config = get_config()
            path = config.get_path('paths.dataset')

        self.path = path
        self._data = None  # "_" indicates a private attribute

    def load(self) -> pd.DataFrame:
        """
        Load the NEO dataset from the configured path.

        Returns:
            pd.DataFrame: Loaded NEO dataset.
        """
        self._data = pd.read_csv(self.path)
        return self._data

    @property  # Access as an attribute
    def data(self) -> pd.DataFrame:
        """
        Get the loaded data. If not loaded yet, load it automatically.

        Returns:
            pd.DataFrame: The loaded dataset.
        """
        if self._data is None:
            self.load()
        return self._data

    def reload(self) -> pd.DataFrame:
        """
        Reload the dataset from the source file.

        Returns:
            pd.DataFrame: Reloaded NEO dataset.
        """
        return self.load()

    def get_summary(self) -> dict:
        """
        Get a summary of the loaded dataset.

        Returns:
            dict: Dictionary containing dataset statistics.
        """
        if self._data is None:
            self.load()

        return {
            'n_rows': self._data.shape[0],
            'n_columns': self._data.shape[1],
            'columns': list(self._data.columns),
            # MB
            'memory_usage': self._data.memory_usage(deep=True).sum() / 1024**2
        }

    def display_summary(self, target_column: str = 'hazardous') -> None:
        """
        Display a formatted summary of the dataset.

        Args:
            target_column (str): Name of the target column to show distribution.
        """
        summary = self.get_summary()
        print(
            f"✓ Data loaded: {summary['n_rows']} rows, {summary['n_columns']} columns")
        print(f"  - Columns: {summary['columns']}")
        print(f"  - Memory usage: {summary['memory_usage']:.2f} MB")

        if target_column in self._data.columns:
            print(f"  - Distribution of '{target_column}':")
            print(self._data[target_column].value_counts().to_string(
                header=False).replace('\n', '\n    '))
