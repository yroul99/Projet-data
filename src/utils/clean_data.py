# src/utils/clean_data.py
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd

from config import RAW_GAMES, CLEAN_2021, CLEAN_FILE, SEASON, RAW_ARENAS, RAW_ELEV
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
    df = df[df["season"] == SEASON].drop_duplicates("id", keep="first")
    df["team_key"] = df["home_team"].map(_normalize)  # clé pour joindre l'arène du domicile
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
        get = lambda k: (b.get(k) or {}).get("value")
        team, arena = get("teamLabel"), get("arenaLabel")
        if not team:
            continue
        tk = _normalize(team)
        tk = _normalize(synonyms.get(tk, tk))
        rows.append({
            "team_key": tk,
            "arena": arena,
            "lat": pd.to_numeric(get("lat"), errors="coerce"),
            "lon": pd.to_numeric(get("lon"), errors="coerce"),
            "capacity": pd.to_numeric(get("capacity"), errors="coerce"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["team_key", "arena", "lat", "lon", "capacity"])

    # Une ligne par équipe : priorité aux coords puis à la plus grande capacité
    df["_has_coords"] = df["lat"].notna() & df["lon"].notna()
    df["_cap"] = df["capacity"].fillna(-1)
    df = (df.sort_values(["team_key", "_has_coords", "_cap"], ascending=[True, False, False])
            .drop_duplicates("team_key")
            .drop(columns=["_has_coords", "_cap"]))

    # Overrides optionnels
    ov = Path("data/reference/arenas_overrides.csv")
    if ov.exists():
        o = pd.read_csv(ov)
        o["team_key"] = o["team_key"].map(_normalize)
        df = df.merge(o, on="team_key", how="outer", suffixes=("", "_ov"))
        for c in ["arena", "lat", "lon", "capacity"]:
            oc = c + "_ov"
            if oc in df.columns:
                df[c] = df[oc].combine_first(df[c])
        df = df[["team_key", "arena", "lat", "lon", "capacity"]]

    return df


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
        d[["date", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).sort_values(["team", "date"])

    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days

    home_rest = long.rename(columns={"team": "home_team", "rest_days": "home_rest_days"})[
        ["home_team", "date", "home_rest_days"]
    ]
    away_rest = long.rename(columns={"team": "away_team", "rest_days": "away_rest_days"})[
        ["away_team", "date", "away_rest_days"]
    ]

    d = d.merge(home_rest, on=["home_team", "date"], how="left")
    d = d.merge(away_rest, on=["away_team", "date"], how="left")

    d["home_b2b"] = (d["home_rest_days"] <= 1).fillna(False)
    d["away_b2b"] = (d["away_rest_days"] <= 1).fillna(False)
    d["rest_delta"] = d["home_rest_days"].fillna(3) - d["away_rest_days"].fillna(3)
    return d


# ---------- Pipeline principal ----------
def clean_2021() -> Path:
    df = _read_games_df(RAW_GAMES)

    # Arènes
    if RAW_ARENAS.exists():
        arenas = _read_arenas_json(RAW_ARENAS)
        arenas = _attach_altitude(arenas)   # ajoute elev_m si dispo
        df = df.merge(arenas, on="team_key", how="left")
    else:
        for c in ("arena", "lat", "lon", "capacity"):
            df[c] = None

    # Repos (B2B)
    df = _rest_features(df)

    # Elo + marge attendue neutre + résiduel
    df = run_elo(df, k=20, use_mov=True, start_rating=1500.0)
    df, alpha, b0 = fit_expected_margin_and_residual(df)
    print(f"[ELO] alpha ≈ {alpha:.3f} | intercept b0 ≈ {b0:.3f}")

    # Colonnes à écrire
    base_cols = [
        "date","season","home_team","away_team",
        "home_pts","away_pts","home_diff",
        "arena","lat","lon","capacity","elev_m",
        "home_rest_days","away_rest_days","home_b2b","away_b2b","rest_delta",
        "elo_home_pre","elo_away_pre","elo_delta_pre","elo_exp_home_win",
        "expected_margin_neutral","residual_margin",
    ]
    cols = [c for c in base_cols if c in df.columns]
    df_out = df[cols].copy()

    # Écriture
    CLEAN_2021.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CLEAN_2021, index=False)
    CLEAN_FILE.write_text(CLEAN_2021.read_text())

    coords_ok = df_out[["lat", "lon"]].dropna().shape[0] if {"lat","lon"}.issubset(df_out.columns) else 0
    print(f"[OK] écrit: {CLEAN_2021} et {CLEAN_FILE} — coords non-null lignes = {coords_ok}")
    return CLEAN_2021
