from src.data.loader import NEODataLoader
from src.data.preprocessor import NEODataPreprocessor
from src.utils.model_comparison import ModelManager


def example_optimize_and_compare():
    """Example: Optimize all models then compare performance"""
    print("EXAMPLE : Optimize Then Compare Performance")

    # Load data
    loader = NEODataLoader()
    preprocessor = NEODataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(loader.data)

    # Create manager and add all models
    manager = ModelManager()
    manager.add_models_default()

    # Step 1: Optimize all models using the parameter grids defined in optimization_grids.yaml
    print("\n[STEP 1] Optimizing hyperparameters using configuration from optimization_grids.yaml...")
    print("Note: This may take some time, especially for SVM")
    all_best_params = manager.optimize_all_models(X_train, y_train)

    # Step 2: Compare optimized models
    print("\n[STEP 2] Comparing optimized models...")
    manager.train_and_evaluate_all(X_train, y_train, X_test, y_test, n_runs=3)
    manager.display_summary()
    manager.display_results_table()

    manager.plot_comparison(save_path="results/optimized_model_comparison.png")

    return manager


if __name__ == "__main__":
    example_optimize_and_compare()
