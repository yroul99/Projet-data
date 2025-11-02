# src/utils/clean_data.py
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd

from config import (
    RAW_GAMES, CLEAN_2021, CLEAN_FILE, SEASON,
    RAW_ARENAS, RAW_ELEV
)
from src.utils.elo import run_elo, fit_expected_margin_and_residual


# ---------- Helpers ----------
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower())


# ---------- Lecture matchs ----------
def _read_games_df(p: Path) -> pd.DataFrame:
    items = json.loads(p.read_text())
    if isinstance(items, dict) and "data" in items:
        items = items["data"]

    df = pd.json_normalize(items)
    keep = [
        "id", "date", "season",
        "home_team.full_name", "visitor_team.full_name",
        "home_team_score", "visitor_team_score",
    ]
    df = df[keep].rename(columns={
        "home_team.full_name": "home_team",
        "visitor_team.full_name": "away_team",
        "home_team_score": "home_pts",
        "visitor_team_score": "away_pts",
    })

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["home_diff"] = df["home_pts"] - df["away_pts"]
    df["team_key"] = df["home_team"].map(_normalize)

    df = df[df["season"] == SEASON]
    df = df.drop_duplicates(subset="id", keep="first")
    return df


# ---------- Lecture arènes (JSON Wikidata) ----------
def _read_arenas_json(p: Path) -> pd.DataFrame:
    raw = json.loads(p.read_text())
    bindings = (raw.get("results") or {}).get("bindings") or []

    synonyms = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "new orleans hornets": "new orleans pelicans",
        "charlotte bobcats": "charlotte hornets",
    }

    rows = []
    for b in bindings:
        def val(name: str):
            x = b.get(name)
            return None if x is None else x.get("value")

        team = val("teamLabel")
        if team is None:
            continue

        arena = val("arenaLabel")
        cap = val("capacity")
        lat = val("lat")
        lon = val("lon")

        tk = _normalize(team)
        tk = _normalize(synonyms.get(tk, tk))

        rows.append({
            "team_key": tk,
            "arena": arena,
            "lat": pd.to_numeric(lat, errors="coerce"),
            "lon": pd.to_numeric(lon, errors="coerce"),
            "capacity": pd.to_numeric(cap, errors="coerce"),
        })

    if not rows:
        return pd.DataFrame(columns=["team_key", "arena", "lat", "lon", "capacity"])

    df = pd.DataFrame(rows)

    # Choix d'une ligne par équipe : priorité aux coords puis à la capacité
    df["_has_coords"] = df["lat"].notna() & df["lon"].notna()
    df["_cap_rank"] = df["capacity"].fillna(-1)
    df = df.sort_values(["team_key", "_has_coords", "_cap_rank"],
                        ascending=[True, False, False])
    df = df.drop_duplicates(subset="team_key", keep="first")
    df = df.drop(columns=["_has_coords", "_cap_rank"], errors="ignore")

    # Overrides optionnels
    ov_path = Path("data/reference/arenas_overrides.csv")
    if ov_path.exists():
        ov = pd.read_csv(ov_path)
        ov["team_key"] = ov["team_key"].map(_normalize)
        df = df.merge(ov, on="team_key", how="outer", suffixes=("", "_ov"))
        for col in ["arena", "lat", "lon", "capacity"]:
            ov_col = col + "_ov"
            if ov_col in df.columns:
                df[col] = df[ov_col].combine_first(df[col])
                df.drop(columns=[ov_col], inplace=True, errors="ignore")

    return df[["team_key", "arena", "lat", "lon", "capacity"]]


# ---------- Altitude (Open-Elevation, cache JSON) ----------
def _attach_altitude(df_arenas: pd.DataFrame) -> pd.DataFrame:
    if df_arenas is None or df_arenas.empty or not {"lat", "lon"}.issubset(df_arenas.columns):
        return df_arenas

    if not RAW_ELEV.exists():
        coords = (df_arenas[["lat", "lon"]]
                  .dropna().drop_duplicates().itertuples(index=False, name=None))
        from src.utils.get_data import fetch_open_elevation
        fetch_open_elevation(list(coords), force=False)

    elev = pd.DataFrame(json.loads(RAW_ELEV.read_text()))
    if elev.empty:
        return df_arenas
    elev = elev.rename(columns={"latitude": "lat", "longitude": "lon", "elevation": "elev_m"})
    return df_arenas.merge(elev, on=["lat", "lon"], how="left")


# ---------- Repos / B2B ----------
def _rest_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    long = pd.concat([
        d[["date", "home_team"]].rename(columns={"home_team": "team"}),
        d[["date", "away_team"]].rename(columns={"away_team": "team"})
    ], ignore_index=True).sort_values(["team", "date"])

    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days

    home_rest = long.rename(columns={"team": "home_team",
                                     "rest_days": "home_rest_days"})[["home_team", "date", "home_rest_days"]]
    away_rest = long.rename(columns={"team": "away_team",
                                     "rest_days": "away_rest_days"})[["away_team", "date", "away_rest_days"]]

    d = d.merge(home_rest, on=["home_team", "date"], how="left")
    d = d.merge(away_rest, on=["away_team", "date"], how="left")

    d["home_b2b"] = (d["home_rest_days"] <= 1).fillna(False)
    d["away_b2b"] = (d["away_rest_days"] <= 1).fillna(False)
    d["rest_delta"] = d["home_rest_days"].fillna(3) - d["away_rest_days"].fillna(3)
    return d


# ---------- Pipeline principal ----------
from config import RAW_GAMES, RAW_TEAMS, CLEAN_2021, CLEAN_FILE, SEASON, RAW_ARENAS
import pandas as pd
import json
from pathlib import Path
import re

def _normalize_team(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]","", s.lower())

def _read_arenas_df(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # coord sous forme 'Point(lon lat)' → extraire
    if "coord" in df.columns:
        coords = df["coord"].astype(str).str.extract(r'Point\((?P<lon>-?\d+\.?\d*) (?P<lat>-?\d+\.?\d*)\)')
        df["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
    # normalise la clé équipe pour la jointure
    df["team_key"] = df["teamLabel"].astype(str).map(_normalize_team)
    # garde l’essentiel
    keep = ["team_key", "arenaLabel", "lat", "lon", "capacity"]
    for k in keep:
        if k not in df.columns: df[k] = None
    return df[keep].rename(columns={"arenaLabel":"arena"})

def _read_games_df(p: Path) -> pd.DataFrame:
    items = json.loads(p.read_text())
    if isinstance(items, dict) and "data" in items:
        items = items["data"]
    df = pd.json_normalize(items)
    keep = [
        "id","date","season",
        "home_team.full_name","visitor_team.full_name",
        "home_team_score","visitor_team_score"
    ]
    df = df[keep].rename(columns={
        "home_team.full_name":"home_team",
        "visitor_team.full_name":"away_team",
        "home_team_score":"home_pts",
        "visitor_team_score":"away_pts",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["home_diff"] = df["home_pts"] - df["away_pts"]
    df = df[df["season"] == SEASON]
    # clé de jointure normalisée
    df["team_key"] = df["home_team"].astype(str).map(_normalize_team)
    return df

def clean_2021() -> Path:
    df = _read_games_df(RAW_GAMES)

    # Jointure arènes si le CSV RAW existe
    if RAW_ARENAS.exists():
        arenas = _read_arenas_df(RAW_ARENAS)
        df = df.merge(arenas, on="team_key", how="left")
    else:
        # colonnes vides pour que l'app ne casse pas
        for col in ["arena","lat","lon","capacity"]:
            df[col] = None

    # colonnes finales minimalement requises par l’app
    cols = ["date","season","home_team","away_team",
            "home_pts","away_pts","home_diff",
            "arena","lat","lon","capacity"]
    df_out = df[cols].copy()

    CLEAN_2021.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CLEAN_2021, index=False)
    CLEAN_FILE.write_text(CLEAN_2021.read_text())
    print(f"[OK] écrit: {CLEAN_2021} et {CLEAN_FILE}")
    return CLEAN_2021
