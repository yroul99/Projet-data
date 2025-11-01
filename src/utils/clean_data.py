from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from config import RAW_GAMES, CLEAN_2021, CLEAN_FILE, SEASON, RAW_ARENAS

# ---------- Helpers ----------
def _normalize(s: str) -> str:
    """Normalise un nom en clé (minuscule, alphanum, espaces)."""
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
    df["team_key"]  = df["home_team"].map(_normalize)  # clé pour joindre l'arène du HOME

    # Saison + dédup
    df = df[df["season"] == SEASON]
    df = df.drop_duplicates(subset="id", keep="first")

    return df

# ---------- Lecture arènes (robuste à différents schémas) ----------
def _read_arenas_df(p: Path) -> pd.DataFrame:
    """
    Lit le CSV Wikidata des arènes et retourne :
      team_key, arena, lat, lon, capacity
    Tolère deux schémas :
      - lat/lon directs (recommandé via SPARQL)
      - colonne WKT 'coord' = 'Point(lon lat)'
    Si plusieurs lignes par équipe : garde celle ayant coordonnées + capacité max.
    """
    df = pd.read_csv(p)

    # Normalisation lat/lon
    has_latlon = {"lat", "lon"}.issubset(df.columns)
    if not has_latlon and "coord" in df.columns:
        coords = df["coord"].astype(str).str.extract(
            r"Point\((?P<lon>-?\d+\.?\d*) (?P<lat>-?\d+\.?\d*)\)"
        )
        df["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
        has_latlon = True
    if not has_latlon:
        # crée colonnes vides si Wikidata ne renvoie pas de coord
        df["lat"] = pd.NA
        df["lon"] = pd.NA

    # Capacité en numérique si présente
    if "capacity" in df.columns:
        df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    else:
        df["capacity"] = pd.NA

    # Clé d'équipe normalisée
    # La requête SPARQL renvoie en général 'teamLabel' et 'arenaLabel'
    if "teamLabel" not in df.columns or "arenaLabel" not in df.columns:
        # Sécurités : on essaie des alternatives si jamais
        # (mais normalement avec notre SPARQL on a ces colonnes)
        team_col  = next((c for c in df.columns if "team"  in c.lower() and "label" in c.lower()), None)
        arena_col = next((c for c in df.columns if "arena" in c.lower() and "label" in c.lower()), None)
        if team_col is None or arena_col is None:
            raise ValueError("Colonnes d'étiquettes d'équipe/salle manquantes dans le CSV Wikidata.")
        df["teamLabel"]  = df[team_col]
        df["arenaLabel"] = df[arena_col]

    df["team_key"] = df["teamLabel"].map(_normalize)

    # Choix d'une ligne par équipe :
    # 1) avec coordonnées (True d'abord), 2) puis capacité décroissante
    df["_has_coords"] = df["lat"].notna() & df["lon"].notna()
    df["_cap_rank"]   = df["capacity"].fillna(-1)

    df = df.sort_values(["team_key", "_has_coords", "_cap_rank"], ascending=[True, False, False])
    df = df.drop_duplicates(subset="team_key", keep="first")

    # Nettoie les colonnes techniques
    df = df.drop(columns=["_has_coords", "_cap_rank"], errors="ignore")

    # Schéma de sortie
    out = df[["team_key", "arenaLabel", "lat", "lon", "capacity"]].rename(
        columns={"arenaLabel": "arena"}
    )

    # Overrides (optionnels) : data/reference/arenas_overrides.csv
    ov_path = Path("data/reference/arenas_overrides.csv")
    if ov_path.exists():
        ov = pd.read_csv(ov_path)
        ov["team_key"] = ov["team_key"].map(_normalize)
        # Merge prioritaire : override > wikidata
        out = out.merge(ov, on="team_key", how="outer", suffixes=("", "_ov"))
        for col in ["arena", "lat", "lon", "capacity"]:
            if col + "_ov" in out.columns:
                out[col] = out[col + "_ov"].combine_first(out[col])
                out = out.drop(columns=[col + "_ov"], errors="ignore")

    # Types
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["capacity"] = pd.to_numeric(out["capacity"], errors="coerce")

    return out

# ---------- Clean principal ----------
def clean_2021() -> Path:
    df = _read_games_df(RAW_GAMES)

    # Jointure des arènes (si le fichier existe)
    if RAW_ARENAS.exists():
        arenas = _read_arenas_df(RAW_ARENAS)
        df = df.merge(arenas, on="team_key", how="left")
    else:
        for c in ("arena", "lat", "lon", "capacity"):
            df[c] = None

    # Colonnes finales (carte + histogramme)
    cols = [
        "date", "season",
        "home_team", "away_team",
        "home_pts", "away_pts", "home_diff",
        "arena", "lat", "lon", "capacity",
    ]
    df_out = df[cols].copy()

    # Sauvegarde
    CLEAN_2021.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CLEAN_2021, index=False)
    CLEAN_FILE.write_text(CLEAN_2021.read_text())

    # Petit log utile
    coords_ok = df_out[["lat", "lon"]].dropna().shape[0]
    print(f"[OK] écrit: {CLEAN_2021} et {CLEAN_FILE} — coords non-null lignes = {coords_ok}")

    return CLEAN_2021
