"""
Main script demonstrating the complete ML pipeline:
1. Load data with NEODataLoader
2. Preprocess with NEODataPreprocessor (fit_transform vs transform)
3. Train a Decision Tree model
"""

from src.data.loader import NEODataLoader
from src.data.preprocessor import NEODataPreprocessor
from src.models.tree_decision_classifier import DecisionTreeModel
from src.config import get_config


def main():
    config = get_config()

    print("=" * 60)
    print("DEMONSTRATION - NEO CLASSIFICATION")
    print("=" * 60)

    # STEP 1: DATA LOADING
    print("\n[1] Loading data...")
    loader = NEODataLoader()
    df = loader.data
    loader.display_summary(target_column=config.get_param('preprocessing.target_column', 'hazardous'))
    
    # STEP 2: PREPROCESSING
    print("\n[2] Preprocessing data...")
    preprocessor = NEODataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    
    # Use display methods
    preprocessor.display_split_info()
    preprocessor.display_normalization_stats()


if __name__ == "__main__":
    main()
