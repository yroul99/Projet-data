"""Pipeline de nettoyage : dédoublonne, enrichit (arènes/Elo) et écrit les jeux."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import importlib
import logging
import pandas as pd

import config as cfg
from src.utils.elo import run_elo, fit_expected_margin_and_residual

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("clean_data")


def CFG():
    """Recharge config pour honorer DATA_DIR/SEASON fixés par les tests."""
    return importlib.reload(cfg)


# --------------------------------------------------------------------------------------
# Déduplication & helpers
# --------------------------------------------------------------------------------------
DEDUP_KEYS_CANDIDATES: List[List[str]] = [
    ["game_id"],  # idéal si dispo
    ["date", "home_team", "away_team"],
]


def _pick_dedup_keys(df: pd.DataFrame) -> List[str]:
    """Retourne la meilleure clé dispo pour l'unicité des matches."""
    for keys in DEDUP_KEYS_CANDIDATES:
        if all(k in df.columns for k in keys):
            return keys
    return []


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les types critiques (date → 'YYYY-MM-DD' au format string)."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def ensure_home_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Crée la marge brute si absente."""
    df = df.copy()
    if "home_diff" not in df.columns and {"home_pts", "away_pts"}.issubset(df.columns):
        df["home_diff"] = df["home_pts"] - df["away_pts"]
    return df


def dedupe_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les doublons de matches en gardant la ligne la plus « complète ».
    """
    keys = _pick_dedup_keys(df)
    if not keys:
        return df

    df = df.copy()
    nn = df.notna().sum(axis=1)
    df["_nn"] = nn

    if "date" in keys:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    df = df.sort_values(by=keys + ["_nn"], ascending=[True] * len(keys) + [False])
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="first").drop(columns="_nn")
    after = len(df)
    log.info("[clean_data] dedupe %s -> %s using keys=%s", before, after, keys)
    return df


def _normalize(s: str) -> str:
    """Retourne une clé normalisée (minuscules alphanumériques + espaces)."""
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower())


# --------------------------------------------------------------------------------------
# Lecture matchs (balldontlie JSON)
# --------------------------------------------------------------------------------------
def _read_games_df(p: Path) -> pd.DataFrame:
    """Charge le JSON balldontlie en ne gardant que la saison courante CFG."""
    C = CFG()
    log.info("Reading games from %s", p)

    items = json.loads(p.read_text())
    if isinstance(items, dict) and "data" in items:
        items = items["data"]

    df = pd.json_normalize(items)
    keep = [
        "id",
        "date",
        "season",
        "home_team.full_name",
        "visitor_team.full_name",
        "home_team_score",
        "visitor_team_score",
    ]
    df = df[keep].rename(
        columns={
            "home_team.full_name": "home_team",
            "visitor_team.full_name": "away_team",
            "home_team_score": "home_pts",
            "visitor_team_score": "away_pts",
        }
    )
    # date au format date (compatible to_datetime en aval)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["game_id"] = df["id"]
    df["home_diff"] = df["home_pts"] - df["away_pts"]
    df = df[df["season"] == C.SEASON].drop_duplicates("game_id", keep="first")

    # clés normalisées pour joindre aux arènes
    df["team_key"] = df["home_team"].map(_normalize)  # domicile
    df["away_team_key"] = df["away_team"].map(_normalize)  # extérieur
    return df


# --------------------------------------------------------------------------------------
# Lecture arènes (Wikidata SPARQL JSON)
# --------------------------------------------------------------------------------------
def _read_arenas_json(p: Path) -> pd.DataFrame:
    """Transforme l'export SPARQL Wikidata en une ligne par franchise."""
    raw = json.loads(p.read_text())
    bindings = (raw.get("results") or {}).get("bindings") or []

    synonyms = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "new orleans hornets": "new orleans pelicans",
        "charlotte bobcats": "charlotte hornets",
    }

    def get_value(binding: dict, key: str) -> str | None:
        """Extrait la valeur d'un binding SPARQL"""
        return (binding.get(key) or {}).get("value")

    rows = []
    for b in bindings:
        team, arena = get_value(b, "teamLabel"), get_value(b, "arenaLabel")
        lat = get_value(b, "lat")
        lon = get_value(b, "lon")
        capacity = get_value(b, "capacity")
        if not team:
            continue
        tk = _normalize(team)
        tk = _normalize(synonyms.get(tk, tk))
        rows.append(
            {
                "team_key": tk,
                "arena": arena,
                "lat": pd.to_numeric(lat, errors="coerce"),
                "lon": pd.to_numeric(lon, errors="coerce"),
                "capacity": pd.to_numeric(capacity, errors="coerce"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["team_key", "arena", "lat", "lon", "capacity"])

    # Une ligne par équipe : priorité aux coords puis à la plus grande capacité
    df["_has_coords"] = df["lat"].notna() & df["lon"].notna()
    df["_cap"] = df["capacity"].fillna(-1)
    df = (
        df.sort_values(
            ["team_key", "_has_coords", "_cap"], ascending=[True, False, False]
        )
        .drop_duplicates("team_key")
        .drop(columns=["_has_coords", "_cap"])
    )

    # Overrides optionnels (facultatif)
    C = CFG()
    ov = C.DATA_DIR / "reference" / "arenas_overrides.csv"
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


# --------------------------------------------------------------------------------------
# Altitude (Open-Elevation, cache JSON)
# --------------------------------------------------------------------------------------
def _attach_altitude(df_arenas: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute la colonne elev_m en joignant sur lat/lon avec petit arrondi
    pour éviter les ratés d'égalité flottante.
    """
    if (
        df_arenas is None
        or df_arenas.empty
        or not {"lat", "lon"}.issubset(df_arenas.columns)
    ):
        return df_arenas

    C = CFG()

    # Crée le cache si absent
    if not C.RAW_ELEV.exists():
        log.warning("RAW_ELEV absent; attempting fetch_open_elevation(...)")
        coords = (
            df_arenas[["lat", "lon"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        from src.utils.get_data import fetch_open_elevation

        fetch_open_elevation(list(coords), force=False)

    # Charge cache
    elev = pd.DataFrame(json.loads(C.RAW_ELEV.read_text()))
    if elev.empty:
        return df_arenas

    df = df_arenas.copy()
    df["lat_r"] = df["lat"].astype(float).round(5)
    df["lon_r"] = df["lon"].astype(float).round(5)

    elev["lat_r"] = elev["latitude"].astype(float).round(5)
    elev["lon_r"] = elev["longitude"].astype(float).round(5)
    elev = elev.rename(columns={"elevation": "elev_m"})

    out = df.merge(
        elev[["lat_r", "lon_r", "elev_m"]], on=["lat_r", "lon_r"], how="left"
    )
    return out.drop(columns=["lat_r", "lon_r"])


# --------------------------------------------------------------------------------------
# Repos / B2B
# --------------------------------------------------------------------------------------
def _rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule repos et indicateurs B2B pour équipes domicile/extérieur."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])

    long = pd.concat(
        [
            d[["date", "home_team"]].rename(columns={"home_team": "team"}),
            d[["date", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).sort_values(["team", "date"])

    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days

    home_rest = long.rename(
        columns={"team": "home_team", "rest_days": "home_rest_days"}
    )[["home_team", "date", "home_rest_days"]]
    away_rest = long.rename(
        columns={"team": "away_team", "rest_days": "away_rest_days"}
    )[["away_team", "date", "away_rest_days"]]

    d = d.merge(home_rest, on=["home_team", "date"], how="left")
    d = d.merge(away_rest, on=["away_team", "date"], how="left")

    d["home_b2b"] = (d["home_rest_days"] <= 1).fillna(False)
    d["away_b2b"] = (d["away_rest_days"] <= 1).fillna(False)
    d["rest_delta"] = d["home_rest_days"].fillna(3) - d["away_rest_days"].fillna(3)
    return d


# --------------------------------------------------------------------------------------
# Pipeline principal
# --------------------------------------------------------------------------------------
def clean_2021() -> Path:
    """Exécute toute la pipeline de nettoyage/enrichissement pour la saison."""
    C = CFG()  # chemins basés sur DATA_DIR/SEASON courants

    df = _read_games_df(C.RAW_GAMES)

    # 1) Arènes : JSON SPARQL si dispo, sinon fallback CSV overrides
    arenas_df = None
    if C.RAW_ARENAS.exists():
        arenas_df = _read_arenas_json(C.RAW_ARENAS)
    else:
        # Fallback CSV dans le DATA_DIR de travail des tests
        ov = C.DATA_DIR / "reference" / "arenas_overrides.csv"
        if ov.exists():
            a = pd.read_csv(ov)
            # colonnes minimales attendues
            for col in ("team_key", "arena", "lat", "lon", "capacity"):
                if col not in a.columns:
                    a[col] = pd.NA
            a["team_key"] = a["team_key"].map(_normalize)
            arenas_df = a[["team_key", "arena", "lat", "lon", "capacity"]]

    if arenas_df is not None and not arenas_df.empty:
        # Si lat/lon présents, on attache (ou recharge) l'altitude depuis le cache
        if {"lat", "lon"}.issubset(arenas_df.columns):
            arenas_df = _attach_altitude(arenas_df)

        # jointure DOMICILE (team_key)
        df = df.merge(arenas_df, on="team_key", how="left")
        # duplique pour compat dashboard
        df["home_elev_m"] = df.get("elev_m")
        df["home_capacity"] = df.get("capacity")

        # jointure EXTERIEUR (away_team_key) pour récupérer away_elev_m
        away_cols = arenas_df.rename(
            columns={
                "team_key": "away_key",
                "elev_m": "away_elev_m",
                "capacity": "away_capacity",
                "lat": "away_lat",
                "lon": "away_lon",
                "arena": "away_arena",
            }
        )
        keep_away = [
            c
            for c in ["away_key", "away_elev_m", "away_capacity"]
            if c in away_cols.columns
        ]
        df = df.merge(
            away_cols[keep_away],
            left_on="away_team_key",
            right_on="away_key",
            how="left",
        )
        if "away_key" in df.columns:
            df = df.drop(columns=["away_key"])

        # Δ altitude (domicile – extérieur)
        if {"home_elev_m", "away_elev_m"}.issubset(df.columns):
            df["delta_elev"] = df["home_elev_m"] - df["away_elev_m"]
        else:
            df["delta_elev"] = pd.NA
    else:
        # colonnes attendues par le dashboard/tests si pas d'arènes du tout
        for c in (
            "arena",
            "lat",
            "lon",
            "capacity",
            "elev_m",
            "home_elev_m",
            "away_elev_m",
            "delta_elev",
            "home_capacity",
            "away_capacity",
        ):
            df[c] = None

    # 2) Repos / B2B
    df = _rest_features(df)

    # 3) Elo + marge attendue neutre + résiduel
    df = run_elo(df, k=20, use_mov=True, start_rating=1500.0)
    df, alpha, b0 = fit_expected_margin_and_residual(df)
    log.info("[ELO] alpha ≈ %.3f | intercept b0 ≈ %.3f", alpha, b0)

    # 4) Déduplication matches AVANT l'écriture finale
    df = dedupe_matches(df)

    # 5) Colonnes à écrire
    base_cols = [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_pts",
        "away_pts",
        "home_diff",
        "arena",
        "lat",
        "lon",
        "capacity",
        "elev_m",
        "home_elev_m",
        "away_elev_m",
        "delta_elev",
        "home_capacity",
        "away_capacity",
        "home_rest_days",
        "away_rest_days",
        "home_b2b",
        "away_b2b",
        "rest_delta",
        "elo_home_pre",
        "elo_away_pre",
        "elo_delta_pre",
        "elo_exp_home_win",
        "expected_margin_neutral",
        "residual_margin",
    ]
    cols = [c for c in base_cols if c in df.columns]
    df_out = df[cols].copy()

    # 6) Écritures (dans le DATA_DIR actuel)
    C.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(C.CLEAN_2021, index=False)
    Path(C.CLEAN_FILE).write_text(Path(C.CLEAN_2021).read_text(), encoding="utf-8")

    coords_ok = (
        df_out[["lat", "lon"]].notna().all().all()
        if {"lat", "lon"}.issubset(df_out.columns)
        else False
    )
    log.info(
        "[OK] écrit: %s et %s — lat/lon complets = %s",
        C.CLEAN_2021,
        C.CLEAN_FILE,
        coords_ok,
    )

    # 7) Summary (utilise le même DATA_DIR)
    try:
        from scripts.save_summary import main as save_sum

        save_sum()
    except Exception as e:
        log.warning("[WARN] summary skipped: %s", e)

    # 8) Métadonnées
    meta = {
        "season": int(C.SEASON),
        "sources": {
            "games": "balldontlie …",
            "arenas": "Wikidata SPARQL / overrides CSV",
            "elevation_cache": str(C.RAW_ELEV),
        },
    }
    (C.CLEANED_DIR / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info("Wrote cleaned dataset to %s", C.CLEAN_2021)
    return C.CLEAN_2021
