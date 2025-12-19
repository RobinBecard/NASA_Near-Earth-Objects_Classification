from src.data.loader import NEODataLoader
from src.data.preprocessor import NEODataPreprocessor
from src.utils.model_comparison import ModelManager


def evaluate_all_models():
    print("EVALUATING ALL MODELS\n")

    loader = NEODataLoader()
    preprocessor = NEODataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(loader.data)
    preprocessor.display_split_info()

    manager = ModelManager()
    manager.add_models_default()

    while True:
        try:
            n_runs_input = input(
                "\nNumber of runs for each model (recommended: 3-5): ").strip()
            n_runs = int(n_runs_input)
            if n_runs > 0:
                break
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nEvaluation cancelled.")
            return None

    print(f"\nTRAINING AND EVALUATION ({n_runs} run(s) per model)\n")

    results = manager.train_and_evaluate_all(
        X_train, y_train, X_test, y_test, n_runs=n_runs
    )

    print("\nRESULTS\n")
    manager.display_summary()
    print()
    df_results = manager.display_results_table()

    print()
    try:
        save_plot = input(
            "Do you want to save the comparison plot? (y/n): ").strip().lower()
        if save_plot in ['y', 'yes', 'o', 'oui']:
            save_path = "results/model_comparison.png"
            manager.plot_comparison(save_path=save_path)
            print(f"\nPlot saved: {save_path}")
        else:
            manager.plot_comparison()
    except KeyboardInterrupt:
        print("\nDisplaying plot...")
        manager.plot_comparison()

    print("\nEVALUATION COMPLETED\n")

    return results, df_results


if __name__ == "__main__":
    evaluate_all_models()
