from pathlib import Path
import os

# Dossiers
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DATA_RAW   = DATA_DIR / "raw"
DATA_CLEAN = DATA_DIR / "cleaned"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_CLEAN.mkdir(parents=True, exist_ok=True)

# Endpoints publics
ENDPOINTS = {
    "bl_games":  "https://api.balldontlie.io/v1/games",
    "bl_teams":  "https://api.balldontlie.io/v1/teams",
    "wd_sparql": "https://query.wikidata.org/sparql",
    # (pour plus tard si tu ajoutes Elo/altitude)
    # "elo_538_csv": "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv",
    # "openmeteo_elev": "https://api.open-meteo.com/v1/elevation",
}

# Fichiers RAW
RAW_GAMES   = DATA_RAW / "balldontlie_games_2021.json"
RAW_TEAMS   = DATA_RAW / "balldontlie_teams.json"
RAW_ARENAS  = DATA_RAW / "wikidata_arenas.csv"
# (option futurs fetch)
# RAW_ELO     = DATA_RAW / "nba_elo.csv"
# RAW_ELEV    = DATA_RAW / "arenas_elevation.json"

# Fichiers CLEAN
CLEAN_2021  = DATA_CLEAN / "dataset_clean_2021.csv"
CLEAN_FILE  = DATA_CLEAN / "dataset_clean.csv"

# Saison choisie
SEASON = 2021  # 2021-22

# Clé API balldontlie (mettre dans l'env PowerShell : $env:BALLDONTLIE_API_KEY="...")
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", None)
