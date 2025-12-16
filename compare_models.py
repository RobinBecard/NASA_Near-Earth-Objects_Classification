from src.data.loader import NEODataLoader
from src.utils.model_comparison import ModelComparator


def main():
    # Load data
    loader = NEODataLoader()
    loader.display_summary()

    # Run full comparison
    comparator = ModelComparator()
    comparator.run_full_comparison(
        df=loader.data,
        n_runs=3,
        save_plot="results/model_comparison.png"
    )


if __name__ == "__main__":
    main()
