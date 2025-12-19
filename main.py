"""
Main script demonstrating the complete ML pipeline:
1. Load data with NEODataLoader
2. Preprocess with NEODataPreprocessor (fit_transform vs transform)
3. Train a Logistic Regression model
"""

from src.data.loader import NEODataLoader
from src.data.preprocessor import NEODataPreprocessor
from src.data.loader import NEODataLoader
from src.utils.model_comparison import ModelManager
from src.models.tree_decision_classifier import DecisionTreeClassifier
from src.config import get_config


def main():
    config = get_config()

    # print("=" * 60)
    # print("DEMONSTRATION - NEO CLASSIFICATION")
    # print("=" * 60)

    # # STEP 1: DATA LOADING
    # print("\n[1] Loading data...")
    # loader = NEODataLoader()
    # df = loader.data
    # loader.display_summary(target_column=config.get_param(
    #     'preprocessing.target_column', 'hazardous'))

    # # STEP 2: PREPROCESSING
    # print("\n[2] Preprocessing data...")
    # preprocessor = NEODataPreprocessor()
    # X_train, X_test, y_train, y_test = preprocessor.preprocess(df)

    # # Use display methods
    # preprocessor.display_split_info()
    # preprocessor.display_normalization_stats()

    # ##################################################
    # ## EXAMPLE OF MODEL TRAINING WITH DECISION TREE ##
    # # STEP 3: MODEL TRAINING
    # print("\n[3] Training Decision Tree model...")

    # model = DecisionTreeClassifier(
    #     name="Decision Tree Classifier",
    #     params={}  # Use default values from config.yaml
    # )

    # model.train(X_train, y_train)
    # print("✓ Model trained successfully!")

    # # STEP 4: EVALUATION
    # print("\n[4] Evaluating model...")
    # metrics = model.evaluate(X_test, y_test)
    # model.display_metrics(metrics)

    # #############################################################
    # ## EXAMPLE OF HYPERPARAMETER OPTIMIZATION WITH GRID SEARCH ##
    # # STEP 5: HYPERPARAMETER OPTIMIZATION
    # print("\n[5] Optimizing hyperparameters with Grid Search...")
    # param_grid = {
    #     'max_depth': [5, 10, 15],
    #     'criterion': ['gini', 'entropy']
    # }

    # best_params = model.optimize(X_train, y_train, param_grid)

    # print("✓ Hyperparameter optimization completed!")
    # print("Best Parameters found:")
    # for param, value in best_params.items():
    #     print(f"  • {param}: {value}")
    # # Re-evaluate with optimized model
    # print("\nRe-evaluating optimized model...")
    # optimized_metrics = model.evaluate(X_test, y_test)
    # model.display_metrics(optimized_metrics)

    # print("\n" + "=" * 60)

    # Load data
    loader = NEODataLoader()
    loader.display_summary()

    # Run full comparison
    manager = ModelManager()
    manager.run_full_comparison(
        df=loader.data,
        n_runs=3,
        save_plot="results/model_comparison.png"
    )


if __name__ == "__main__":
    main()
