"""Utilitaires d'ingestion: balldontlie, Wikidata et Open-Elevation."""
from __future__ import annotations
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import requests

from config import (
    ENDPOINTS,
    RAW_GAMES,
    RAW_TEAMS,
    RAW_ARENAS,
    RAW_ELEV,
    SEASON,
    BALLDONTLIE_API_KEY,
)

UA = {"User-Agent": "esiee-projet-data/1.0"}


def _fresh_enough(path: Path, hours: int = 24) -> bool:
    """Renvoie True si le fichier existe et reste dans la fenêtre de fraîcheur."""
    return path.exists() and (
        datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        < timedelta(hours=hours)
    )


def _save_json(path: Path, data) -> None:
    """Écrit une charge JSON sérialisable en UTF-8."""
    path.write_text(json.dumps(data, ensure_ascii=False))


def _auth_headers() -> Dict[str, str]:
    """Ajoute la clé API balldontlie si fournie dans l'environnement."""
    h = UA.copy()
    if BALLDONTLIE_API_KEY:
        # balldontlie expects the raw key, not "Bearer ..."
        h["Authorization"] = BALLDONTLIE_API_KEY
    return h


def _get_with_backoff(
    url: str, *, params: Dict[str, Any], headers: Dict[str, str], retries: int = 6
):
    """GET with exponential backoff; handles HTTP 429 via Retry-After if present."""
    delay = 1.0
    for _ in range(retries):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            wait_s = int(ra) + 1 if ra and ra.isdigit() else max(2, int(delay))
            time.sleep(wait_s)
            delay = min(delay * 2, 30)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"Too many 429 responses on {url} with {params}")


# ---------- balldontlie (teams) ----------
def fetch_balldontlie_teams(force: bool = False) -> Path:
    """Télécharge et met en cache la liste statique des franchises NBA."""
    if _fresh_enough(RAW_TEAMS) and not force:
        return RAW_TEAMS
    r = _get_with_backoff(ENDPOINTS["bl_teams"], params={}, headers=_auth_headers())
    _save_json(RAW_TEAMS, r.json())
    return RAW_TEAMS


# ---------- balldontlie (regular season 2021-22 with cursor pagination) ----------
def fetch_balldontlie_games_2021(force: bool = False, postseason: bool = False) -> Path:
    """
    NBA 2021-22 (regular season) monthly windows + cursor pagination.
    Resumes from existing RAW, dedups by id, snapshots after each page.
    """
    RAW_GAMES.parent.mkdir(parents=True, exist_ok=True)
    url = ENDPOINTS["bl_games"]
    headers = _auth_headers()

    # Resume from existing file if any
    all_games, seen = [], set()
    if RAW_GAMES.exists():
        try:
            existing = json.loads(RAW_GAMES.read_text())
            if isinstance(existing, dict) and "data" in existing:
                existing = existing["data"]
            for g in existing:
                gid = g.get("id")
                if gid is not None and gid not in seen:
                    seen.add(gid)
                    all_games.append(g)
            print(f"[resume] RAW existant : {len(all_games)} matchs chargés")
        except Exception:
            pass

    windows = [
        ("2021-10-01", "2021-11-01"),
        ("2021-11-01", "2021-12-01"),
        ("2021-12-01", "2022-01-01"),
        ("2022-01-01", "2022-02-01"),
        ("2022-02-01", "2022-03-01"),
        ("2022-03-01", "2022-04-01"),
        ("2022-04-01", "2022-05-01"),
    ]

    for start, end in windows:
        cursor = None
        while True:
            params = {
                "seasons[]": SEASON,
                "per_page": 100,
                "postseason": str(postseason).lower(),
                "start_date": start,
                "end_date": end,
            }
            if cursor:
                params["cursor"] = cursor

            r = _get_with_backoff(url, params=params, headers=headers)
            payload = r.json()
            data = payload.get("data") or []
            meta = payload.get("meta") or {}

            new_count = 0
            for g in data:
                gid = g.get("id")
                if gid not in seen:
                    seen.add(gid)
                    all_games.append(g)
                    new_count += 1

            _save_json(RAW_GAMES, all_games)
            print(
                f"[{start}→{end}] +{new_count} (total {len(all_games)}) "
                f"cursor={meta.get('next_cursor')}"
            )

            cursor = meta.get("next_cursor")
            if not data or not cursor:
                break

            time.sleep(0.4)

    _save_json(RAW_GAMES, all_games)
    print(f"[done] total RAW = {len(all_games)}")
    return RAW_GAMES


# --- WIKIDATA (arènes NBA) en JSON ---


def fetch_wikidata_arenas(force: bool = False) -> Path:
    """Exécute la requête SPARQL et met en cache le JSON des arènes."""
    if RAW_ARENAS.exists() and not force:
        return RAW_ARENAS

    query = """
    SELECT ?team ?teamLabel ?arena ?arenaLabel ?lat ?lon ?capacity WHERE {
      ?team wdt:P118 wd:Q155223; wdt:P115 ?arena.
      ?arena p:P625 ?coordStmt.
      ?coordStmt psv:P625 ?coordNode.
      ?coordNode wikibase:geoLatitude ?lat ;
                 wikibase:geoLongitude ?lon .
      OPTIONAL { ?arena wdt:P1083 ?capacity. }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    r = requests.get(
        ENDPOINTS["wd_sparql"],
        params={"format": "json", "query": query},
        headers=UA,
        timeout=60,
    )
    r.raise_for_status()
    RAW_ARENAS.parent.mkdir(parents=True, exist_ok=True)  # <-- AJOUT
    RAW_ARENAS.write_text(json.dumps(r.json(), ensure_ascii=False))
    return RAW_ARENAS


# ---------- Open-Elevation ----------
def fetch_open_elevation(
    coords: List[Tuple[float, float]], force: bool = False
) -> Path:
    """
    Fetch elevations for list of (lat, lon); cache in RAW_ELEV as list of dicts:
    [{latitude, longitude, elevation}, ...]
    """
    RAW_ELEV.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if RAW_ELEV.exists() and not force:
        try:
            existing = json.loads(RAW_ELEV.read_text())
        except Exception:
            existing = []
    have = {
        (round(e["latitude"], 6), round(e["longitude"], 6))
        for e in existing
        if all(k in e for k in ("latitude", "longitude"))
    }

    # normalize & dedupe
    todo = []
    for lat, lon in coords:
        key = (round(float(lat), 6), round(float(lon), 6))
        if key not in have:
            todo.append({"latitude": key[0], "longitude": key[1]})

    if not todo:
        return RAW_ELEV

    url = ENDPOINTS["open_elevation"]
    out = existing[:]
    for i in range(0, len(todo), 100):
        chunk = {"locations": todo[i : i + 100]}
        r = requests.post(url, json=chunk, timeout=60)
        r.raise_for_status()
        out.extend(r.json().get("results", []))
        time.sleep(0.2)

    RAW_ELEV.write_text(json.dumps(out, ensure_ascii=False))
    return RAW_ELEV


# ---------- Aggregator ----------
def get_raw(force: bool = False) -> dict[str, Path]:
    """Surcouche pratique pour rafraîchir tous les artefacts bruts."""
    paths = {}
    paths["teams"] = fetch_balldontlie_teams(force=force)
    paths["games"] = fetch_balldontlie_games_2021(force=force, postseason=False)
    try:
        paths["arenas"] = fetch_wikidata_arenas(force=force)
    except Exception as e:
        print("[WARN] wikidata fetch failed:", e)
    return paths
