from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from config import RAW_GAMES, CLEAN_2021, CLEAN_FILE, SEASON, RAW_ARENAS

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]","", str(s).lower())

def _read_games_df(p: Path) -> pd.DataFrame:
    items = json.loads(p.read_text())
    if isinstance(items, dict) and "data" in items:
        items = items["data"]
    df = pd.json_normalize(items)

    keep = [
        "id","date","season",
        "home_team.full_name","visitor_team.full_name",
        "home_team_score","visitor_team_score",
    ]
    df = df[keep].rename(columns={
        "home_team.full_name":"home_team",
        "visitor_team.full_name":"away_team",
        "home_team_score":"home_pts",
        "visitor_team_score":"away_pts",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["home_diff"] = df["home_pts"] - df["away_pts"]
    df["team_key"] = df["home_team"].map(_normalize)  # clé pour joindre l'arène du "home"
    df = df.drop_duplicates(subset="id", keep="first")
    df = df[df["season"] == SEASON]
    return df

def _read_arenas_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # extraire lat/lon depuis 'Point(lon lat)'
    coords = df["coord"].astype(str).str.extract(
        r"Point\((?P<lon>-?\d+\.?\d*) (?P<lat>-?\d+\.?\d*)\)"
    )
    df["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
    df["capacity"] = pd.to_numeric(df.get("capacity"), errors="coerce")
    df["team_key"] = df["teamLabel"].map(_normalize)
    return df[["team_key","arenaLabel","lat","lon","capacity"]].rename(
        columns={"arenaLabel":"arena"}
    )

def clean_2021() -> Path:
    df = _read_games_df(RAW_GAMES)

    if RAW_ARENAS.exists():
        arenas = _read_arenas_df(RAW_ARENAS)
        df = df.merge(arenas, on="team_key", how="left")
    else:
        for c in ("arena","lat","lon","capacity"):
            df[c] = None

    cols = [
        "date","season","home_team","away_team",
        "home_pts","away_pts","home_diff",
        "arena","lat","lon","capacity"
    ]
    df_out = df[cols].copy()

    CLEAN_2021.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CLEAN_2021, index=False)
    CLEAN_FILE.write_text(CLEAN_2021.read_text())
    print(f"[OK] écrit: {CLEAN_2021} et {CLEAN_FILE}")
    return CLEAN_2021
