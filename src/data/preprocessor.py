import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import get_config

class NEODataPreprocessor:
    """
    Class responsible for preprocessing the NEO dataset for modeling.
    """
    def __init__(
        self, 
        numerical_features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: Optional[bool] = None
    ):
        """
        Initialize the NEO Data Preprocessor.

        Args:
            numerical_features (List[str] | None): List of numerical feature names.
            target_column (str | None): Name of the target column.
            test_size (float | None): Proportion of the dataset to include in the test split.
            random_state (int | None): Random state for reproducibility.
            stratify (bool | None): Whether to stratify the train-test split.
        """
        config = get_config()
        
        self.numerical_features = numerical_features or config.get_param(
            'preprocessing.numerical_features',
            ['est_diameter_min', 'est_diameter_max', 'relative_velocity', 
             'miss_distance', 'absolute_magnitude']
        )
        self.target_column = target_column or config.get_param('preprocessing.target_column', 'hazardous')
        self.test_size = test_size if test_size is not None else config.get_param('preprocessing.test_size', 0.2)
        self.random_state = random_state if random_state is not None else config.get_param('preprocessing.random_state', 42)
        self.stratify = stratify if stratify is not None else config.get_param('preprocessing.stratify', True)
        
        self._preprocessor = None
        self._is_fitted = False
        
        # Store the split data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline.

        Returns:
            ColumnTransformer: The preprocessing pipeline.
        """
        return ColumnTransformer([
            ('num', StandardScaler(), self.numerical_features),
        ])
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into training and testing sets.

        Args:
            df (pd.DataFrame): Raw NEO dataset.

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        X = df[self.numerical_features]
        y = df[self.target_column]
        
        stratify_param = y if self.stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train: pd.DataFrame) -> 'NEODataPreprocessor':
        """
        Fit the preprocessor on training data.

        Args:
            X_train (pd.DataFrame): Training features.

        Returns:
            NEODataPreprocessor: Self for method chaining.
        """
        if self._preprocessor is None:
            self._preprocessor = self._create_preprocessor()
        
        self._preprocessor.fit(X_train)
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.

        Args:
            X (pd.DataFrame): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        
        Raises:
            ValueError: If preprocessor is not fitted yet.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        return self._preprocessor.transform(X)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessor and transform the data.

        Args:
            X (pd.DataFrame): Data to fit and transform.

        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X)
        return self.transform(X)
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline: split, fit, and transform.

        Args:
            df (pd.DataFrame): Raw NEO dataset.

        Returns:
            Tuple: (X_train, X_test, y_train, y_test) - preprocessed data.
        """
        # Split the data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Fit and transform
        X_train_processed = self.fit_transform(X_train)
        X_test_processed = self.transform(X_test)
        
        # Store for later access
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names used by the preprocessor.

        Returns:
            List[str]: List of feature names.
        """
        return self.numerical_features.copy()
    
    def get_split_summary(self) -> dict:
        """
        Get a summary of the train-test split.

        Returns:
            dict: Dictionary containing split statistics.
        
        Raises:
            ValueError: If data has not been split yet.
        """
        if self.X_train is None:
            raise ValueError("Data has not been split yet. Call preprocess() first.")
        
        return {
            'n_train': self.X_train.shape[0],
            'n_test': self.X_test.shape[0],
            'n_features': self.X_train.shape[1],
            'test_size_ratio': self.test_size,
            'train_distribution': self.y_train.value_counts().to_dict(),
            'test_distribution': self.y_test.value_counts().to_dict(),
            'train_stats': {
                'min': float(self.X_train.min()),
                'max': float(self.X_train.max()),
                'mean': float(self.X_train.mean()),
                'std': float(self.X_train.std())
            },
            'test_stats': {
                'min': float(self.X_test.min()),
                'max': float(self.X_test.max()),
                'mean': float(self.X_test.mean()),
                'std': float(self.X_test.std())
            }
        }
    
    def display_split_info(self) -> None:
        """
        Display formatted information about the train-test split.
        """
        summary = self.get_split_summary()
        
        print(f"✓ Data split:")
        print(f"  - Train: {summary['n_train']} samples")
        print(f"  - Test:  {summary['n_test']} samples")
        print(f"  - Number of features: {summary['n_features']}")
        print(f"  - Features: {self.get_feature_names()}")
    
    def display_normalization_stats(self) -> None:
        """
        Display normalization statistics for train and test sets.
        """
        summary = self.get_split_summary()
        
        print("\n Normalization statistics:")
        train_stats = summary['train_stats']
        test_stats = summary['test_stats']
        
        print(f"  - X_train: min={train_stats['min']:.3f}, max={train_stats['max']:.3f}, "
              f"mean={train_stats['mean']:.3f}, std={train_stats['std']:.3f}")
        print(f"  - X_test:  min={test_stats['min']:.3f}, max={test_stats['max']:.3f}, "
              f"mean={test_stats['mean']:.3f}, std={test_stats['std']:.3f}")
        print("  → Data is centered around 0 with std ≈ 1")

