from pathlib import Path
import os

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DATA_RAW   = DATA_DIR / "raw"
DATA_CLEAN = DATA_DIR / "cleaned"
for p in (DATA_RAW, DATA_CLEAN):
    p.mkdir(parents=True, exist_ok=True)

# Saison 2021-22 (balldontlie utilise 2021 pour la régulière 2021-22)
SEASON = 2021

# Endpoints publics
ENDPOINTS = {
    "bl_games": "https://api.balldontlie.io/v1/games",
    "bl_teams": "https://api.balldontlie.io/v1/teams",
}

# Fichiers RAW (inchangés)
RAW_GAMES = DATA_RAW / f"balldontlie_games_{SEASON}.json"
RAW_TEAMS = DATA_RAW / "balldontlie_teams.json"

# Fichiers CLEAN (consommés par l'app)
CLEAN_2021 = DATA_CLEAN / "dataset_clean_2021.csv"
CLEAN_FILE = DATA_CLEAN / "dataset_clean.csv"

# Clé API (si tu en as une) : set dans PowerShell =>  $env:BALLDONTLIE_API_KEY="ta_cle"
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", None)
