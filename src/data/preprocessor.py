import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.config import get_config

class NEODataPreprocessor:
    """
    Class responsible for preprocessing the NEO dataset for modeling.
    Handles log-transformation, scaling, and train/test splitting.
    """
    def __init__(
        self, 
        log_features: Optional[List[str]] = None,
        other_features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: Optional[bool] = None
    ):
        """
        Initialize the NEO Data Preprocessor.

        Args:
            log_features (List[str] | None): Features requiring log transformation (skewed distributions).
            other_features (List[str] | None): Features requiring only scaling (no log transformation).
            target_column (str | None): Name of the target column (y).
            test_size (float | None): Proportion of the dataset to include in the test split.
            random_state (int | None): Random state for reproducibility.
            stratify (bool | None): Whether to stratify the train-test split based on target.
        """
        config = get_config()
        
        # Features requiring Log transformation (skewed distributions)
        self.log_features = log_features if log_features is not None else config.get_param(
            'preprocessing.log_features',
            ['est_diameter_max', 'relative_velocity', 'miss_distance']
        )
        # Features requiring standard scaling only
        self.other_features = other_features if other_features is not None else config.get_param(
            'preprocessing.other_features',
            ['absolute_magnitude']
        )

        self.target_column = target_column or config.get_param('preprocessing.target_column', 'hazardous')
        self.test_size = test_size if test_size is not None else config.get_param('preprocessing.test_size', 0.2)
        self.random_state = random_state if random_state is not None else config.get_param('preprocessing.random_state', 42)
        self.stratify = stratify if stratify is not None else config.get_param('preprocessing.stratify', True)
        
        self._preprocessor = None
        self._is_fitted = False
        
        # Placeholders for data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline using ColumnTransformer.
        
        Strategies:
        1. Log-features: Apply log1p then RobustScaler.
        2. Other-features: Apply RobustScaler directly.
        
        Note: We use RobustScaler instead of StandardScaler to handle outliers better.
        """
        # Pipeline for skewed features: Log -> RobustScaler
        log_pipeline = Pipeline([
            ('log', FunctionTransformer(np.log1p, validate=False)),
            ('scaler', RobustScaler()) 
        ])
        
        # Pipeline for normal features: RobustScaler
        std_pipeline = Pipeline([
            ('scaler', RobustScaler())
        ])
        
        return ColumnTransformer([
            ('log_scaled', log_pipeline, self.log_features),
            ('robust_scaled', std_pipeline, self.other_features)
        ], remainder='drop')

    def fit(self, X_train: pd.DataFrame) -> 'NEODataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X_train (pd.DataFrame): Training features.
        
        Returns:
            self: The fitted preprocessor instance.
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
            np.ndarray: Transformed numpy array.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        return self._preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform the data in one step.
        """
        self.fit(X)
        return self.transform(X)

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Execute the full preprocessing workflow:
        1. Separate Target/Features
        2. Split Train/Test
        3. Fit on Train
        4. Transform Train & Test
        
        Args:
            df (pd.DataFrame): The raw dataframe.
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        # 1. Separation
        X = df.drop(columns=[self.target_column], errors='ignore')
        y = df[self.target_column]
        
        # 2. Splitting
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y if self.stratify else None
        )
        
        # 3. & 4. Fitting & Transforming
        self.X_train = self.fit_transform(X_train_raw)
        self.X_test = self.transform(X_test_raw)
        
        self.y_train = y_train
        self.y_test = y_test
        
        self.display_split_info()
        self.display_normalization_stats()
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_names(self) -> List[str]:
        """Return the list of processed features in order: [log_features, other_features]."""
        return self.log_features + self.other_features

    def get_split_summary(self) -> Dict:
        """Return a dictionary of split statistics."""
        if self.X_train is None:
            return {}
        return {
            'n_train': self.X_train.shape[0],
            'n_test': self.X_test.shape[0],
            'n_features': self.X_train.shape[1],
            'train_stats': {
                'min': float(self.X_train.min()), 'max': float(self.X_train.max()),
                'mean': float(self.X_train.mean()), 'std': float(self.X_train.std()),
                'median': float(np.median(self.X_train))
            },
            'test_stats': {
                'min': float(self.X_test.min()), 'max': float(self.X_test.max()),
                'mean': float(self.X_test.mean()), 'std': float(self.X_test.std()),
                'median': float(np.median(self.X_test))
            }
        }
    
    def display_split_info(self) -> None:
        """Print summary of the data split."""
        summary = self.get_split_summary()
        if not summary: print("No data processed."); return
        print(f"✓ Data split: Train={summary['n_train']}, Test={summary['n_test']}, Features={summary['n_features']}")
        print(f"  Features list: {self.get_feature_names()}")

    def display_normalization_stats(self) -> None:
        """Print validation stats for normalization."""
        summary = self.get_split_summary()
        if not summary: return
        tr = summary['train_stats']
        print(f"\nNormalization check (RobustScaler - Median centered around 0):")
        print(f"  Train: median={tr['median']:.3f}, mean={tr['mean']:.3f}, std={tr['std']:.3f}")