from __future__ import annotations
import json, time, os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from config import ENDPOINTS, RAW_GAMES, RAW_TEAMS, SEASON, BALLDONTLIE_API_KEY

UA = {"User-Agent": "esiee-projet-data/1.0"}

def _fresh_enough(path: Path, hours: int = 24) -> bool:
    return path.exists() and (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < timedelta(hours=hours))

def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False))

def _auth_headers() -> Dict[str, str]:
    h = UA.copy()
    if BALLDONTLIE_API_KEY:
        # balldontlie attend la clé brute (pas "Bearer ...")
        h["Authorization"] = BALLDONTLIE_API_KEY
    return h

def _get_with_backoff(url: str, *, params: Dict[str, Any], headers: Dict[str, str], retries: int = 6):
    """GET avec backoff (gère 429 via Retry-After s’il est présent)."""
    delay = 1.0
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            wait_s = int(ra) + 1 if ra and ra.isdigit() else max(2, int(delay))
            time.sleep(wait_s)
            delay = min(delay * 2, 30)  # expo backoff, plafonné
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"Too many 429 responses on {url} with {params}")

def fetch_balldontlie_teams(force: bool = False) -> Path:
    if _fresh_enough(RAW_TEAMS) and not force:
        return RAW_TEAMS
    r = _get_with_backoff(ENDPOINTS["bl_teams"], params={}, headers=_auth_headers())
    _save_json(RAW_TEAMS, r.json())
    return RAW_TEAMS

def fetch_balldontlie_games_2021(force: bool = False, postseason: bool = False) -> Path:
    if _fresh_enough(RAW_GAMES) and not force:
        return RAW_GAMES

    url = ENDPOINTS["bl_games"]
    all_games = []
    page = 1
    headers = _auth_headers()

    while True:
        params = {
            "seasons[]": SEASON,
            "per_page": 100,
            "page": page,
            "postseason": str(postseason).lower(),
        }
        r = _get_with_backoff(url, params=params, headers=headers)
        payload = r.json()
        data = payload.get("data") or []
        if not data:
            break
        all_games.extend(data)

        # on persiste la progression pour éviter de tout perdre en cas de 429/plantage
        _save_json(RAW_GAMES, all_games)

        page += 1
        time.sleep(1.2)  # sois gentil avec le quota

    # une dernière sauvegarde “propre”
    _save_json(RAW_GAMES, all_games)
    return RAW_GAMES

def get_raw(force: bool = False) -> dict[str, Path]:
    paths = {}
    paths["teams"] = fetch_balldontlie_teams(force=force)
    paths["games"] = fetch_balldontlie_games_2021(force=force, postseason=False)
    return paths
