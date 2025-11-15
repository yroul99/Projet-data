"""Tests unitaires du pipeline de nettoyage"""
import pandas as pd
import pytest


def test_rest_features_b2b_flag():
    """Vérifie que _rest_features détecte correctement les B2B"""
    from src.utils.clean_data import _rest_features

    df = pd.DataFrame(
        {
            "date": ["2021-10-20", "2021-10-21", "2021-10-25"],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "D"],
            "season": [2021, 2021, 2021],
            "home_pts": [1, 1, 1],
            "away_pts": [0, 0, 0],
        }
    )

    out = _rest_features(df)
    assert {"home_b2b", "away_b2b", "rest_delta"}.issubset(out.columns)

    # Le match du 21 est B2B pour A - utiliser == au lieu de is
    assert out.loc[out["date"] == "2021-10-21", "home_b2b"].iloc[0]


def test_dedupe_keeps_most_complete_row():
    """Vérifie que dedupe_matches garde la ligne la plus complète"""
    from src.utils.clean_data import dedupe_matches, normalize_types

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
    assert out.iloc[0]["y"] == 2  # Garde la ligne non-null
