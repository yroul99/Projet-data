from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from config import RAW_GAMES, RAW_TEAMS, CLEAN_2021, CLEAN_FILE, SEASON

def _read_games_df(p: Path) -> pd.DataFrame:
    items = json.loads(p.read_text())
    if isinstance(items, dict) and "data" in items:  # sécurité si format différent
        items = items["data"]
    df = pd.json_normalize(items)

    # Colonnes utiles
    keep = [
        "id","date","season",
        "home_team.full_name","visitor_team.full_name",
        "home_team_score","visitor_team_score",
        # facultatif: "postseason"
    ]
    df = df[keep].rename(columns={
        "home_team.full_name":"home_team",
        "visitor_team.full_name":"away_team",
        "home_team_score":"home_pts",
        "visitor_team_score":"away_pts",
    })
    # Normalisations
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["home_diff"] = df["home_pts"] - df["away_pts"]
    df = df[df["season"] == SEASON]  # garde 2021-22
    return df

def clean_2021() -> Path:
    df = _read_games_df(RAW_GAMES)
    # garde un schéma propre pour l'app (même si pas encore de carte)
    cols = ["date","season","home_team","away_team","home_pts","away_pts","home_diff"]
    df_out = df[cols].copy()
    CLEAN_2021.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CLEAN_2021, index=False)

    # alias consommé par l'app
    CLEAN_FILE.write_text(CLEAN_2021.read_text())
    print(f"[OK] écrit: {CLEAN_2021.name} et {CLEAN_FILE.name}")
    return CLEAN_2021
