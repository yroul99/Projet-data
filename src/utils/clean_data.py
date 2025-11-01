from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from config import RAW_GAMES, CLEAN_2021, CLEAN_FILE, SEASON, RAW_ARENAS

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
    df["team_key"]  = df["home_team"].map(_normalize)

    df = df[df["season"] == SEASON]
    df = df.drop_duplicates(subset="id", keep="first")
    return df

# ---------- Lecture arènes (JSON Wikidata) ----------
def _read_arenas_json(p: Path) -> pd.DataFrame:
    raw = json.loads(p.read_text())
    bindings = (raw.get("results") or {}).get("bindings") or []

    # alias pour harmoniser avec balldontlie (qui expose 'Los Angeles Clippers')
    synonyms = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        # vieux alias au cas où
        "new orleans hornets": "new orleans pelicans",
    }

    rows = []
    for b in bindings:
        def val(name:str):
            x = b.get(name)
            return None if x is None else x.get("value")

        team  = val("teamLabel")
        arena = val("arenaLabel")
        cap   = val("capacity")
        lat   = val("lat")
        lon   = val("lon")

        tk = _normalize(team)
        tk = _normalize(synonyms.get(tk, tk))

        rows.append({
            "team_key": tk,
            "arena": arena,
            "lat": pd.to_numeric(lat, errors="coerce"),
            "lon": pd.to_numeric(lon, errors="coerce"),
            "capacity": pd.to_numeric(cap, errors="coerce"),
        })

    df = pd.DataFrame(rows)

    # Si plusieurs entrées par équipe : priorise coords puis capacité
    if not df.empty:
        df["_has_coords"] = df["lat"].notna() & df["lon"].notna()
        df["_cap_rank"]   = df["capacity"].fillna(-1)
        df = df.sort_values(["team_key","_has_coords","_cap_rank"], ascending=[True, False, False])
        df = df.drop_duplicates("team_key", keep="first")
        df = df.drop(columns=["_has_coords","_cap_rank"], errors="ignore")

    # overrides manuels (optionnels)
    ov_path = Path("data/reference/arenas_overrides.csv")
    if ov_path.exists():
        ov = pd.read_csv(ov_path)
        ov["team_key"] = ov["team_key"].map(_normalize)
        df = df.merge(ov, on="team_key", how="outer", suffixes=("", "_ov"))
        for col in ["arena","lat","lon","capacity"]:
            if col + "_ov" in df.columns:
                df[col] = df[col + "_ov"].combine_first(df[col])
                df.drop(columns=[col + "_ov"], inplace=True, errors="ignore")

    return df[["team_key","arena","lat","lon","capacity"]]

# ---------- Clean principal ----------
def clean_2021() -> Path:
    df = _read_games_df(RAW_GAMES)

    if RAW_ARENAS.exists():
        arenas = _read_arenas_json(RAW_ARENAS)
        df = df.merge(arenas, on="team_key", how="left")
    else:
        for c in ("arena","lat","lon","capacity"):
            df[c] = None

cols = [
    "date","season",
    "home_team","away_team",
    "home_pts","away_pts","home_diff",
    "arena","lat","lon","capacity",
    "team_key",   # <-- pour debug/jointures
]
df_out = df[cols].copy()


    CLEAN_2021.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CLEAN_2021, index=False)
    CLEAN_FILE.write_text(CLEAN_2021.read_text())

    coords_ok = df_out[["lat","lon"]].dropna().shape[0]
    print(f"[OK] écrit: {CLEAN_2021} et {CLEAN_FILE} — coords non-null lignes = {coords_ok}")
    return CLEAN_2021
