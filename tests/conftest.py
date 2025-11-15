"""Fixtures pytest communes qui montent un mini dépôt isolé."""
import sys
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# tests/conftest.py


@pytest.fixture()
def tmp_repo(monkeypatch, tmp_path: Path):
    """
    Prépare un mini 'repo' de test isolé (fichiers dans tmp_path) et
    redirige les constantes config.* vers ce répertoire.
    """
    # Arbo "data"
    d_raw = tmp_path / "data" / "raw"
    d_clean = tmp_path / "data" / "cleaned"
    d_ref = tmp_path / "data" / "reference"
    d_raw.mkdir(parents=True, exist_ok=True)
    d_clean.mkdir(parents=True, exist_ok=True)
    d_ref.mkdir(parents=True, exist_ok=True)

    # --- Balldontlie games JSON (2 matches + 1 doublon volontaire) ---
    #   saison 2021, 2 équipes (Lakers/Clippers) avec scores et dates différentes
    games = [
        {
            "id": 1001,
            "date": "2021-10-19",
            "season": 2021,
            "home_team": {"full_name": "Los Angeles Lakers"},
            "visitor_team": {"full_name": "Golden State Warriors"},
            "home_team_score": 114,
            "visitor_team_score": 121,
        },
        {
            "id": 1002,
            "date": "2022-02-03",
            "season": 2021,
            "home_team": {"full_name": "LA Clippers"},
            "visitor_team": {"full_name": "Los Angeles Lakers"},
            "home_team_score": 111,
            "visitor_team_score": 110,
        },
        # doublon exact même id => doit être dédupliqué
        {
            "id": 1002,
            "date": "2022-02-03",
            "season": 2021,
            "home_team": {"full_name": "LA Clippers"},
            "visitor_team": {"full_name": "Los Angeles Lakers"},
            "home_team_score": 111,
            "visitor_team_score": 110,
        },
    ]
    raw_games = d_raw / "balldontlie_games_2021.json"
    raw_games.write_text(json.dumps(games, indent=2))

    # --- Arenas SPARQL-like minimal ---
    arenas_bindings = {
        "results": {
            "bindings": [
                # Lakers (Crypto.com)
                {
                    "teamLabel": {"value": "Los Angeles Lakers"},
                    "arenaLabel": {"value": "Crypto.com Arena"},
                    "lat": {"value": "34.0430"},
                    "lon": {"value": "-118.2673"},
                    "capacity": {"value": "19068"},
                },
                # LA Clippers (même coord à LA pour le test)
                {
                    "teamLabel": {"value": "LA Clippers"},
                    "arenaLabel": {"value": "Crypto.com Arena"},
                    "lat": {"value": "34.0430"},
                    "lon": {"value": "-118.2673"},
                    "capacity": {"value": "19068"},
                },
                # Warriors (Chase Center)
                {
                    "teamLabel": {"value": "Golden State Warriors"},
                    "arenaLabel": {"value": "Chase Center"},
                    "lat": {"value": "37.7680"},
                    "lon": {"value": "-122.3877"},
                    "capacity": {"value": "18064"},
                },
            ]
        }
    }
    raw_arenas = d_raw / "arenas_wikidata.json"
    raw_arenas.write_text(json.dumps(arenas_bindings))

    # --- Arenas overrides CSV (fallback pour clean_2021) ---
    arenas_csv = d_ref / "arenas_overrides.csv"
    arenas_csv_data = pd.DataFrame(
        {
            "team_key": ["los angeles lakers", "la clippers", "golden state warriors"],
            "arena": ["Crypto.com Arena", "Crypto.com Arena", "Chase Center"],
            "lat": [34.0430, 34.0430, 37.7680],
            "lon": [-118.2673, -118.2673, -122.3877],
            "capacity": [19068, 19068, 18064],
        }
    )
    arenas_csv_data.to_csv(arenas_csv, index=False)

    # --- Cache altitude (évite tout réseau) ---
    raw_elev = d_raw / "open_elevation_cache.json"
    # clés arrondies à 6 décimales pour matcher fetch_open_elevation()
    elev_cache = [
        {
            "latitude": 34.043000,
            "longitude": -118.267300,
            "elevation": 89,
        },  # Los Angeles
        {
            "latitude": 37.768000,
            "longitude": -122.387700,
            "elevation": 8,
        },  # San Francisco
    ]
    raw_elev.write_text(json.dumps(elev_cache, indent=2))

    # --- Monkeypatch config vers ce tmp repo ---
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEASON", "2021")

    # Force reimport of config to pick up new env vars
    import importlib
    import config

    importlib.reload(config)

    # Renvoyer les chemins utiles
    return {
        "root": tmp_path,
        "raw_games": raw_games,
        "raw_arenas": raw_arenas,
        "raw_elev": raw_elev,
        "clean_dir": d_clean,
        "clean_file": d_clean / "dataset_clean.csv",
        "summary": d_clean / "summary.json",
    }
