from __future__ import annotations
import json, time, os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import requests

from config import (
    ENDPOINTS,
    RAW_GAMES,
    RAW_TEAMS,
    RAW_ARENAS,   # <-- AJOUT
    SEASON,
    BALLDONTLIE_API_KEY
)

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
            delay = min(delay * 2, 30)  # expo backoff plafonné
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"Too many 429 responses on {url} with {params}")

# ---------- BALLDONTLIE (équipes) ----------
def fetch_balldontlie_teams(force: bool = False) -> Path:
    if _fresh_enough(RAW_TEAMS) and not force:
        return RAW_TEAMS
    r = _get_with_backoff(ENDPOINTS["bl_teams"], params={}, headers=_auth_headers())
    _save_json(RAW_TEAMS, r.json())
    return RAW_TEAMS

# ---------- BALLDONTLIE (matchs régulière 2021-22, pagination par CURSEUR) ----------
def fetch_balldontlie_games_2021(force: bool = False, postseason: bool = False) -> Path:
    """
    NBA 2021-22 (saison régulière) avec fenêtres mensuelles + pagination CURSEUR.
    - Reprend sur un RAW partiel
    - Dédup par id
    - Snapshot après chaque page
    """
    RAW_GAMES.parent.mkdir(parents=True, exist_ok=True)
    url = ENDPOINTS["bl_games"]
    headers = _auth_headers()

    # Reprise si RAW existant
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
    elif not force and RAW_GAMES.exists():
        return RAW_GAMES

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
            print(f"[{start}→{end}] +{new_count} (total {len(all_games)}) cursor={meta.get('next_cursor')}")

            cursor = meta.get("next_cursor")
            if not data or not cursor:
                break

            time.sleep(0.4)  # throttle doux

    _save_json(RAW_GAMES, all_games)
    print(f"[done] total RAW = {len(all_games)}")
    return RAW_GAMES

# ---------- WIKIDATA (arènes NBA) ----------
def fetch_wikidata_arenas(force: bool=False) -> Path:
    """
    Arènes NBA via Wikidata (JSON brut):
      - tenant (P466) = équipe NBA
      - capacité (P1083)
      - lat/lon (via p:P625/psv:P625 → wikibase:geoLatitude/geoLongitude)
    Écrit un JSON brut dans RAW_ARENAS.
    """
    if RAW_ARENAS.exists() and not force:
        return RAW_ARENAS

    q = """
    SELECT ?arena ?arenaLabel ?teamLabel ?capacity ?lat ?lon WHERE {
      ?team wdt:P118 wd:Q155223 .        # league = NBA
      ?arena wdt:P466 ?team .            # tenant (occupant)
      OPTIONAL { ?arena wdt:P1083 ?capacity }   # capacity
      OPTIONAL {
        ?arena p:P625/psv:P625 ?coordNode .
        ?coordNode wikibase:geoLatitude ?lat ;
                   wikibase:geoLongitude ?lon .
      }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    """
    headers = {
        "User-Agent": "esiee-projet-data/1.0",
        "Accept": "application/sparql-results+json",
    }
    params = {
        "query": q,
        "format": "json",
    }
    r = requests.get(ENDPOINTS["wd_sparql"], params=params, headers=headers, timeout=60)
    r.raise_for_status()

    # Sécurité: si jamais ce n'est pas du JSON, on lève une erreur claire
    ctype = r.headers.get("Content-Type","").lower()
    if "json" not in ctype and not r.text.lstrip().startswith("{"):
        raise RuntimeError("Wikidata n'a pas renvoyé du JSON. Réessaie (WDQS rate-limit) ou vérifie les headers.")

    RAW_ARENAS.write_bytes(r.content)
    return RAW_ARENAS




# ---------- Agrégateur ----------
def get_raw(force: bool = False) -> dict[str, Path]:
    paths = {}
    paths["teams"]  = fetch_balldontlie_teams(force=force)
    paths["games"]  = fetch_balldontlie_games_2021(force=force, postseason=False)
    try:
        paths["arenas"] = fetch_wikidata_arenas(force=force)
    except Exception as e:
        print("[WARN] wikidata fetch failed:", e)
    return paths
