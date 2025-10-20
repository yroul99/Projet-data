from pathlib import Path

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DATA_RAW   = DATA_DIR / "raw"
DATA_CLEAN = DATA_DIR / "cleaned"

# Le fichier que l'app lira
CLEAN_FILE = DATA_CLEAN / "dataset_clean.csv"
