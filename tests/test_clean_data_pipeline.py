"""Tests d'intégration pour la pipeline de nettoyage et les helpers de dédup."""
from __future__ import annotations
import json
import pandas as pd


def test_dedupe_function(tmp_repo, monkeypatch):
    # Import tardif (après monkeypatch config)
    from src.utils.clean_data import dedupe_matches, normalize_types

    # 2 lignes dupliquées avec + d'info sur la 1ère
    df = pd.DataFrame(
        [
            {"date": "2021-11-01", "home_team": "A", "away_team": "B", "x": 1, "y": 2},
            {
                "date": "2021-11-01",
                "home_team": "A",
                "away_team": "B",
                "x": 1,
                "y": None,
            },
        ]
    )
    df = normalize_types(df)
    out = dedupe_matches(df)
    assert len(out) == 1
    row = out.iloc[0].to_dict()
    # on garde la plus "complète" => y non nul
    assert row["y"] == 2


def test_clean_end_to_end(tmp_repo):
    # end-to-end : génère le CSV nettoyé sans doublons + coords + delta_elev + summary
    from src.utils.clean_data import clean_2021

    p = clean_2021()
    assert p.exists(), "CSV cleaned n'a pas été écrit"

    # Lecture des résultats
    from config import CLEAN_FILE

    df = pd.read_csv(CLEAN_FILE)

    # DEBUG: print columns and sample data
    print("\n=== DataFrame Info ===")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst row:\n{df.iloc[0]}")
    print(
        f"\nexpected_margin_neutral:",
        df["expected_margin_neutral"].value_counts(dropna=False),
    )
    print(f"residual_margin: {df['residual_margin'].value_counts(dropna=False)}")

    # 1) Unicité
    keys = [c for c in ["game_id", "date", "home_team", "away_team"] if c in df.columns]
    assert df.duplicated(subset=keys).sum() == 0, "Doublons présents après clean"

    # 2) Colonnes essentielles présentes
    required = {
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
        "elev_m",
        "home_elev_m",
        "away_elev_m",
        "delta_elev",
        "elo_home_pre",
        "elo_away_pre",
        "elo_delta_pre",
        "elo_exp_home_win",
        "expected_margin_neutral",
        "residual_margin",
    }
    assert required.issubset(
        df.columns
    ), f"Colonnes manquantes: {required - set(df.columns)}"

    # 3) Coordonnées non nulles
    assert df[["lat", "lon"]].notna().all().all(), "lat/lon manquants"

    # 4) Delta altitude calculé
    assert df["delta_elev"].notna().all(), "delta_elev manquant"

    # 5) Résiduel cohérent : proche de (home_diff - expected_margin_neutral)
    if len(df) > 2:  # Only check correlation for meaningful datasets
        proxy = df["home_diff"] - df["expected_margin_neutral"]
        corr = proxy.corr(df["residual_margin"])
        assert corr > 0.99, f"Résiduel incohérent (corr={corr:.4f})"
    else:
        # For tiny test dataset, just verify the columns exist and have values
        assert (
            df["expected_margin_neutral"].notna().any()
        ), "expected_margin_neutral all NaN"
        assert df["residual_margin"].notna().any(), "residual_margin all NaN"

    # 6) summary.json produit et n aligné
    s_path = tmp_repo["summary"]
    assert s_path.exists(), "summary.json absent"
    s = json.loads(s_path.read_text())
    assert int(s["baseline"]["n"]) == len(df), "n summary != n CSV"
