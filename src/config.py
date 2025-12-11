from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "neo.csv"
RESULTS_DIR = PROJECT_ROOT / "results"