from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from config import RAW_GAMES, CLEAN_2021, CLEAN_FILE, SEASON, RAW_ARENAS

# ---------------- Helpers ----------------
def _normalize(s: str) -> str:
    """Normalise un nom d'équipe en clé : minuscules, alphanum + espaces."""
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower())

def _pick_col(df: pd.DataFrame, patterns_list: list[tuple[str, ...]]) -> str | None:
    """
    Retourne le nom de la première colonne dont le nom contient TOUTES les sous-chaînes
    d'un des tuples de patterns_list (insensible à la casse).
    Ex: patterns_list=[('team','label'),('team','name')] trouvera 'teamLabel' ou 'Team name', etc.
    """
    cols = list(df.columns)
    lower = [c.lower().strip() for c in cols]
    for pats in patterns_list:
        for i, name in enumerate(lower):
            if all(p in name for p in pats):
                return cols[i]
    return None

# ---------------- Lecture matchs ----------------
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

    # clé pour joindre l'arène du domicile
    df["team_key"] = df["home_team"].map(_normalize)

    # Saison + dédup
    df = df[df["season"] == SEASON]
    df = df.drop_duplicates(subset="id", keep="first")
    return df

# ---------------- Lecture arènes (ultra-robuste) ----------------
def _read_arenas_df(p: Path) -> pd.DataFrame:
    """
    Lit le CSV Wikidata et retourne un DF:
      team_key, arena, lat, lon, capacity
    Tolère:
      - lat/lon directs OU une colonne WKT 'coord' = 'Point(lon lat)'
      - noms de colonnes variables: teamLabel/arenaLabel, 'team name', etc.
      - plusieurs lignes par équipe -> garde celle avec coords puis capacité max.
    """
    df = pd.read_csv(p)

    # 1) Identifier colonnes label équipe / arène
    team_col = _pick_col(df, [
        ("team", "label"), ("teamlabel",), ("team", "name"), ("club", "label")
    ]) or _pick_col(df, [("team",)])  # dernier recours
    arena_col = _pick_col(df, [
        ("arena", "label"), ("arenalabel",), ("arena", "name"),
        ("venue", "label"), ("stadium", "label"), ("arena",)
    ])

    if team_col is None or arena_col is None:
        # log minimal pour debug visuel
        print("[WARN] Colonnes disponibles dans wikidata_arenas.csv:", list(df.columns))
        raise ValueError("Impossible d'identifier les colonnes 'teamLabel'/'arenaLabel' (même approchées).")

    # 2) Lat / Lon : soit déjà là, soit via 'coord' en WKT
    lat_col = _pick_col(df, [("lat",), ("latitude",)])
    lon_col = _pick_col(df, [("lon",), ("long",), ("longitude",)])
    if lat_col is None or lon_col is None:
        coord_col = _pick_col(df, [("coord",)])
        if coord_col:
            coords = df[coord_col].astype(str).str.extract(
                r"Point\((?P<lon>-?\d+\.?\d*) (?P<lat>-?\d+\.?\d*)\)"
            )
            df["__lat__"] = pd.to_numeric(coords["lat"], errors="coerce")
            df["__lon__"] = pd.to_numeric(coords["lon"], errors="coerce")
            lat_col, lon_col = "__lat__", "__lon__"
        else:
            # colonnes vides si indisponibles
            df["__lat__"] = pd.NA
            df["__lon__"] = pd.NA
            lat_col, lon_col = "__lat__", "__lon__"

    # 3) Capacité (si présente, sinon NA)
    cap_col = _pick_col(df, [("capacity",), ("seating", "capacity")])
    if cap_col is None:
        df["__cap__"] = pd.NA
        cap_col = "__cap__"

    # 4) Construire la sortie normalisée
    out = pd.DataFrame({
        "team_key": df[team_col].map(_normalize),
        "arena": df[arena_col],
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        "capacity": pd.to_numeric(df[cap_col], errors="coerce"),
    })

    # 5) Choisir 1 ligne par équipe: priorise (coords dispo) puis (capacité max)
    out["_has_coords"] = out["lat"].notna() & out["lon"].notna()
    out["_cap_rank"] = out["capacity"].fillna(-1)
    out = out.sort_values(["team_key", "_has_coords", "_cap_rank"], ascending=[True, False, False])
    out = out.drop_duplicates(subset="team_key", keep="first")
    out = out.drop(columns=["_has_coords", "_cap_rank"], errors="ignore")

    # 6) Overrides manuels (optionnels) : data/reference/arenas_overrides.csv
    ov_path = Path("data/reference/arenas_overrides.csv")
    if ov_path.exists():
        ov = pd.read_csv(ov_path)
        ov["team_key"] = ov["team_key"].map(_normalize)
        out = out.merge(ov, on="team_key", how="outer", suffixes=("", "_ov"))
        for col in ["arena", "lat", "lon", "capacity"]:
            if col + "_ov" in out.columns:
                out[col] = out[col + "_ov"].combine_first(out[col])
                out.drop(columns=[col + "_ov"], inplace=True, errors="ignore")

    return out

# ---------------- Clean principal ----------------
def clean_2021() -> Path:
    df = _read_games_df(RAW_GAMES)

    # Merge arènes si dispo
    if RAW_ARENAS.exists():
        arenas = _read_arenas_df(RAW_ARENAS)
        df = df.merge(arenas, on="team_key", how="left")
    else:
        for c in ("arena", "lat", "lon", "capacity"):
            df[c] = None

    cols = [
        "date", "season",
        "home_team", "away_team",
        "home_pts", "away_pts", "home_diff",
        "arena", "lat", "lon", "capacity",
    ]
    df_out = df[cols].copy()

    CLEAN_2021.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CLEAN_2021, index=False)
    CLEAN_FILE.write_text(CLEAN_2021.read_text())

    coords_ok = df_out[["lat", "lon"]].dropna().shape[0]
    print(f"[OK] écrit: {CLEAN_2021} et {CLEAN_FILE} — coords non-null lignes = {coords_ok}")
    return CLEAN_2021
