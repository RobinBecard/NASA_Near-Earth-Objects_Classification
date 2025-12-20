from src.data.loader import NEODataLoader
from src.data.preprocessor import NEODataPreprocessor
from src.utils.model_comparison import ModelManager


def optimize_all_models_interactive():
    print("OPTIMIZING ALL MODELS\n")

    loader = NEODataLoader()
    preprocessor = NEODataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(loader.data)

    manager = ModelManager()
    manager.add_models_default()

    models_list = list(manager.models.keys())
    print(f"{len(models_list)} models to optimize:")
    for i, model_name in enumerate(models_list, 1):
        print(f"  {i}. {model_name}")

    print("\nSTARTING OPTIMIZATION\n")

    all_optimized_params = {}

    for i, model_name in enumerate(models_list, 1):
        print(f"\n[MODEL {i}/{len(models_list)}] {model_name}")

        while True:
            try:
                response = input(
                    f"Do you want to optimize {model_name}? (y/n/q to quit): ").strip().lower()

                if response in ['q', 'quit']:
                    print("\nOptimization cancelled.")
                    if all_optimized_params:
                        print("\nOptimized parameters so far:")
                        manager._display_all_params_as_yaml(
                            all_optimized_params)
                    return None

                elif response in ['y', 'yes', 'o', 'oui']:
                    print(f"\nOptimizing {model_name}...")

                    best_params = manager.optimize_model(
                        model_name, X_train, y_train)

                    if best_params is not None:
                        all_optimized_params[model_name] = best_params
                        print(f"\n{model_name} successfully optimized")
                    else:
                        print(f"\nOptimization failed for {model_name}")
                    break

                elif response in ['n', 'no', 'non']:
                    print(f"{model_name} skipped")
                    break

                else:
                    print(
                        "Invalid response. Please enter 'y' (yes), 'n' (no) or 'q' (quit)")

            except KeyboardInterrupt:
                print("\nOptimization interrupted.")
                if all_optimized_params:
                    print("\nOptimized parameters so far:")
                    manager._display_all_params_as_yaml(all_optimized_params)
                return None

    print("\nOPTIMIZATION COMPLETED\n")

    if all_optimized_params:
        print(f"{len(all_optimized_params)} model(s) optimized")
        print("\nOptimal parameters for all models:")
        manager._display_all_params_as_yaml(all_optimized_params)
        print("\nCopy these parameters to config.yaml to use them as defaults")
    else:
        print("No models were optimized")

    return all_optimized_params


if __name__ == "__main__":
    optimize_all_models_interactive()
