from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from config import RAW_GAMES, CLEAN_2021, CLEAN_FILE, SEASON, RAW_ARENAS
from src.utils.elo import run_elo, fit_expected_margin_and_residual


# ---------- Helpers ----------
def _normalize(s: str) -> str:
    """Normalise une chaîne en clé (minuscule, alphanum + espaces)."""
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

    # clé pour joindre l'arène du domicile
    df["team_key"] = df["home_team"].map(_normalize)

    # Saison + dédup
    df = df[df["season"] == SEASON]
    df = df.drop_duplicates(subset="id", keep="first")
    return df

# ---------- Lecture arènes (JSON Wikidata) ----------
def _read_arenas_json(p: Path) -> pd.DataFrame:
    """
    Lit le JSON SPARQL (Wikidata) et renvoie:
      team_key, arena, lat, lon, capacity
    Gère quelques alias d'équipes.
    """
    raw = json.loads(p.read_text())
    bindings = (raw.get("results") or {}).get("bindings") or []

    # alias pour harmoniser avec balldontlie (et quelques anciennes dénominations)
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
        arena = val("arenaLabel")
        cap = val("capacity")
        lat = val("lat")
        lon = val("lon")

        if team is None:
            continue

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
        # retourne un DF vide mais au bon schéma
        return pd.DataFrame(columns=["team_key", "arena", "lat", "lon", "capacity"])

    df = pd.DataFrame(rows)

    # Si plusieurs entrées par équipe : priorise coords puis capacité
    df["_has_coords"] = df["lat"].notna() & df["lon"].notna()
    df["_cap_rank"] = df["capacity"].fillna(-1)
    df = df.sort_values(["team_key", "_has_coords", "_cap_rank"],
                        ascending=[True, False, False])
    df = df.drop_duplicates(subset="team_key", keep="first")
    df = df.drop(columns=["_has_coords", "_cap_rank"], errors="ignore")

    # Overrides manuels (optionnels)
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

# ---------- Clean principal ----------
def clean_2021() -> Path:
    df = _read_games_df(RAW_GAMES)

    # Arènes (Wikidata JSON) -> join via team_key
    if RAW_ARENAS.exists():
        arenas = _read_arenas_json(RAW_ARENAS)  # assure-toi que cette fonction existe
        df = df.merge(arenas, on="team_key", how="left")
    else:
        for c in ("arena", "lat", "lon", "capacity"):
            df[c] = None

    # Schéma de sortie + features Elo
    cols = [
        "date","season",
        "home_team","away_team",
        "home_pts","away_pts","home_diff",
        "arena","lat","lon","capacity",
        "team_key",
    ]
    df_out = df[cols].copy()

    # --- Elo (terrain neutre) + attendu + résiduel AVANT d'écrire ---
    df_out = run_elo(df_out, k=20, use_mov=True, start_rating=1500.0)
    df_out, alpha = fit_expected_margin_and_residual(df_out)
    df_out["expected_margin_neutral"] = df_out["expected_margin_neutral"].round(2)
    df_out["residual_margin"] = df_out["residual_margin"].round(2)
    print(f"[ELO] alpha ≈ {alpha:.3f}")

    # Écriture CSV (enrichi)
    CLEAN_2021.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CLEAN_2021, index=False)
    CLEAN_FILE.write_text(CLEAN_2021.read_text())

    coords_ok = df_out[["lat", "lon"]].dropna().shape[0]
    print(f"[OK] écrit: {CLEAN_2021} et {CLEAN_FILE} — coords non-null lignes = {coords_ok}")
    return CLEAN_2021

    


