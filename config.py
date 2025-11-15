"""Chemins et réglages centraux partagés par la pipeline et le dashboard."""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

# Saison ciblée (les tests peuvent la remplacer via l'env)
SEASON = int(os.getenv("SEASON", "2021"))

# Répertoire racine des données (surgeable pour les tests)
DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()

RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
REFERENCE_DIR = DATA_DIR / "reference"

# On s'assure de disposer des dossiers utilisés par la pipeline
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers sources/produits utilisés par la pipeline
RAW_GAMES = RAW_DIR / f"balldontlie_games_{SEASON}.json"
RAW_ARENAS = RAW_DIR / "wikidata_arenas.json"
RAW_ELEV = RAW_DIR / "open_elevation_cache.json"

CLEAN_2021 = CLEANED_DIR / f"dataset_clean_{SEASON}.csv"
CLEAN_FILE = CLEANED_DIR / "dataset_clean.csv"
SUMMARY_JSON = CLEANED_DIR / "summary.json"

# API endpoints and credentials (required by get_data.py)
_BALLDONTLIE_BASE = "https://api.balldontlie.io/api/v1"
ENDPOINTS = {
    "bl_base": _BALLDONTLIE_BASE,
    "bl_teams": f"{_BALLDONTLIE_BASE}/teams",
    "bl_games": f"{_BALLDONTLIE_BASE}/games",
    "open_elevation": "https://api.open-elevation.com/api/v1/lookup",
    "wd_sparql": "https://query.wikidata.org/sparql",
}

RAW_TEAMS = RAW_DIR / "balldontlie_teams.json"

BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "")
