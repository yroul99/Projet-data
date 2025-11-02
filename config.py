# config.py
from pathlib import Path
import os

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
DATA_RAW   = DATA_DIR / "raw"
DATA_CLEAN = DATA_DIR / "cleaned"
for p in (DATA_RAW, DATA_CLEAN):
    p.mkdir(parents=True, exist_ok=True)

ENDPOINTS = {
    "bl_games":  "https://api.balldontlie.io/v1/games",
    "bl_teams":  "https://api.balldontlie.io/v1/teams",
    "wd_sparql": "https://query.wikidata.org/sparql",
    "open_elevation": "https://api.open-elevation.com/api/v1/lookup",
}

RAW_GAMES  = DATA_RAW / "balldontlie_games_2021.json"
RAW_TEAMS  = DATA_RAW / "balldontlie_teams.json"
RAW_ARENAS = DATA_RAW / "wikidata_arenas.json"
RAW_ELEV   = DATA_RAW / "open_elevation_cache.json"   # <<--- nouveau

CLEAN_2021 = DATA_CLEAN / "dataset_clean_2021.csv"
CLEAN_FILE = DATA_CLEAN / "dataset_clean.csv"

SEASON = 2021
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY")
