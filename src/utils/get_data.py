from __future__ import annotations
import json, time, os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import requests
from config import ENDPOINTS, RAW_GAMES, RAW_TEAMS, SEASON, BALLDONTLIE_API_KEY

UA = {"User-Agent": "esiee-projet-data/1.0"}

def _fresh_enough(path: Path, hours: int = 24) -> bool:
    """True si le fichier existe et date de moins de 'hours' heures."""
    return path.exists() and (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < timedelta(hours=hours))

def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False))

def fetch_balldontlie_teams(force: bool = False) -> Path:
    if _fresh_enough(RAW_TEAMS) and not force:
        return RAW_TEAMS
    headers = UA.copy()
    if BALLDONTLIE_API_KEY:
        headers["Authorization"] = f"Bearer {87b6c405-cb91-4566-a530-a5f92945adbe}"
    r = requests.get(ENDPOINTS["bl_teams"], headers=headers, timeout=30)
    r.raise_for_status()
    _save_json(RAW_TEAMS, r.json())
    return RAW_TEAMS

def fetch_balldontlie_games_2021(force: bool = False, postseason: bool = False) -> Path:
    """Télécharge tous les matchs de la saison régulière 2021-22 (season=2021)."""
    if _fresh_enough(RAW_GAMES) and not force:
        return RAW_GAMES

    headers = UA.copy()
    if BALLDONTLIE_API_KEY:
        headers["Authorization"] = f"Bearer {87b6c405-cb91-4566-a530-a5f92945adbe}"

    url = ENDPOINTS["bl_games"]
    all_games = []
    page = 1
    while True:
        params = {
            "seasons[]": SEASON,
            "per_page": 100,
            "page": page,
            "postseason": str(postseason).lower(),  # 'false' pour régulière
        }
        r = requests.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data") or []
        if not data:
            break
        all_games.extend(data)
        page += 1
        time.sleep(0.25)  # gentil avec l'API
    _save_json(RAW_GAMES, all_games)
    return RAW_GAMES

def get_raw(force: bool = False) -> dict[str, Path]:
    """Point d'entrée: télécharge TEAMS + GAMES 2021-22 (RAW inchangé)."""
    paths = {}
    paths["teams"] = fetch_balldontlie_teams(force=force)
    paths["games"] = fetch_balldontlie_games_2021(force=force, postseason=False)
    return paths
