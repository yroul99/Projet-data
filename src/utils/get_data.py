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
    """
    Télécharge TOUTE la régulière 2021-22 (NBA) :
    - Fenêtres mensuelles (limite les 429)
    - Reprise sur RAW partiel (si déjà présent)
    - Dédup par id
    - Snapshot après chaque page
    """
    from config import ENDPOINTS, RAW_GAMES, SEASON
    import json, time, requests

    RAW_GAMES.parent.mkdir(parents=True, exist_ok=True)

    url = ENDPOINTS["bl_games"]
    headers = _auth_headers()

    # ---- reprise sur RAW existant (même si force=True) ----
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
        # cas normal : on réutilise si frais (géré ailleurs par _fresh_enough)
        return RAW_GAMES

    # ---- fenêtres mensuelles ----
    windows = [
        ("2021-10-01", "2021-11-01"),
        ("2021-11-01", "2021-12-01"),
        ("2021-12-01", "2022-01-01"),
        ("2022-01-01", "2022-02-01"),
        ("2022-02-01", "2022-03-01"),
        ("2022-03-01", "2022-04-01"),
        ("2022-04-01", "2022-05-01"),
    ]

    for (start, end) in windows:
        page = 1
        while True:
            params = {
                "seasons[]": SEASON,
                "per_page": 100,
                "page": page,
                "postseason": str(postseason).lower(),
                "start_date": start,
                "end_date": end,
            }
            r = _get_with_backoff(url, params=params, headers=headers)
            payload = r.json()
            data = payload.get("data") or []
            if not data:
                print(f"[{start}→{end}] page {page}: 0 items → break")
                break

            new_count = 0
            for g in data:
                gid = g.get("id")
                if gid not in seen:
                    seen.add(gid)
                    all_games.append(g)
                    new_count += 1

            # snapshot progression
            _save_json(RAW_GAMES, all_games)

            meta = payload.get("meta") or {}
            cur = meta.get("current_page") or page
            tot = meta.get("total_pages") or cur
            print(f"[{start}→{end}] page {cur}/{tot}: +{new_count} (total {len(all_games)})")

            if cur >= tot:
                break
            page += 1
            time.sleep(0.6)

    _save_json(RAW_GAMES, all_games)
    print(f"[done] total RAW = {len(all_games)}")
    return RAW_GAMES


def get_raw(force: bool = False) -> dict[str, Path]:
    paths = {}
    paths["teams"] = fetch_balldontlie_teams(force=force)
    paths["games"] = fetch_balldontlie_games_2021(force=force, postseason=False)
    return paths
