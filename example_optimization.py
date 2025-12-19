

from src.data.loader import NEODataLoader
from src.data.preprocessor import NEODataPreprocessor
from src.utils.model_comparison import ModelManager


def example_optimize_and_compare():
    """Example: Optimize all models then compare performance (excluding SVM for speed)"""
    print("EXAMPLE : Optimize Then Compare Performance")

    # Load data
    loader = NEODataLoader()
    preprocessor = NEODataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(loader.data)

    # Create manager and add models (excluding SVM for speed)
    manager = ModelManager()
    from src.models.tree_decision_classifier import DecisionTreeModel
    from src.models.logistic_regression_classifier import LogisticRegressionClassifier
    from src.models.naive_bayes_classifier import NaiveBayesClassifier
    from src.models.nn_classifier import NNClassifier
    from src.models.least_squares_classifier import LeastSquaresClassifier

    manager.add_model(DecisionTreeModel(name="Decision Tree", params={}))
    manager.add_model(LogisticRegressionClassifier(
        name="Logistic Regression", params={}))
    manager.add_model(NaiveBayesClassifier(name="Naive Bayes", params={}))
    manager.add_model(NNClassifier(name="Neural Network", params={}))
    manager.add_model(LeastSquaresClassifier(name="Least Squares", params={}))

    # Override optimization grids with minimal grids for fast optimization
    # Note: SVM excluded because it's too slow for a quick example
    manager.optimization_grids['param_grids'] = {
        'decision_tree': {
            'max_depth': [10, 20],
            'criterion': ['gini']
        },
        'logistic_regression': {
            'C': [1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['liblinear']
        },
        'naive_bayes': {
            'var_smoothing': [1e-9, 1e-8]
        },
        'neural_network': {
            'hidden_layer_sizes': [[64], [128]],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001],
            'learning_rate_init': [0.001]
        },
        'least_squares': {
            'alpha': [1.0, 10.0],
            'solver': ['auto']
        }
    }

    # Override optimization config for more verbosity
    manager.optimization_grids['optimization_config']['verbose'] = 2

    # Step 1: Optimize all models with minimal grids
    print("\n[STEP 1] Optimizing hyperparameters with minimal grids...")
    print("Note: SVM excluded from this example due to long execution time")
    all_best_params = manager.optimize_all_models(X_train, y_train, cv=3)

    # Step 2: Compare optimized models
    print("\n[STEP 2] Comparing optimized models...")
    manager.train_and_evaluate_all(X_train, y_train, X_test, y_test, n_runs=3)
    manager.display_summary()
    manager.display_results_table()

    manager.plot_comparison(save_path="results/optimized_model_comparison.png")

    return manager


def example_custom_param_grid():
    """Example: Use a custom parameter grid"""
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Custom Parameter Grid")
    print("="*80)

    # Load and preprocess data
    loader = NEODataLoader()
    preprocessor = NEODataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(loader.data)

    # Create manager and add models
    manager = ModelManager()
    manager.add_models_default()

    # Define a custom, smaller param grid for faster testing
    custom_grid = {
        'max_depth': [5, 10],
        'criterion': ['gini']
    }

    # Optimize with custom grid
    best_params = manager.optimize_model(
        "Decision Tree",
        X_train,
        y_train,
        param_grid=custom_grid
    )

    print(f"\nBest parameters with custom grid: {best_params}")

    return manager


if __name__ == "__main__":
    # Uncomment the example you want to run

    # Example 1: Optimize a single model
    # example_optimize_single_model()

    # Example 2: Optimize all models
    # example_optimize_all_models()

    # Example 3: Optimize then compare (RECOMMENDED)
    example_optimize_and_compare()

    # Example 4: Use custom parameter grid
    # example_custom_param_grid()
