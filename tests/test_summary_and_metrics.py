"""Valide le contenu de summary.json et l'absence de doublons."""
from __future__ import annotations
import json
import pandas as pd


def test_save_summary_outputs(tmp_repo):
    """save_summary doit émettre les sections baseline/elo avec des stats."""
    # Force reimport to use correct tmp paths
    import importlib
    import config

    importlib.reload(config)

    # Nettoyage & summary
    from src.utils.clean_data import clean_2021

    clean_2021()

    from scripts.save_summary import main as save_summary

    save_summary()

    s = json.loads(tmp_repo["summary"].read_text())

    # Clés principales présentes
    for k in ["baseline", "elo_linear"]:
        assert k in s, f"Section {k} manquante"

    base = s["baseline"]
    elo = s["elo_linear"]

    # Types numériques attendus
    assert isinstance(base["n"], int)
    assert isinstance(base["home_diff_mean"], (int, float))
    assert isinstance(elo["b0"], (int, float))
    assert isinstance(elo["alpha"], (int, float))

    # IC95 existent (2 bornes)
    assert len(base["home_diff_ci95"]) == 2
    assert len(elo["b0_ci95"]) == 2
    assert len(elo["alpha_ci95"]) == 2


def test_no_duplicates_again(tmp_repo):
    """clean_2021 doit produire un dataset sans doublons de matches."""
    from src.utils.clean_data import clean_2021

    clean_2021()

    from config import CLEAN_FILE

    df = pd.read_csv(CLEAN_FILE)
    keys = [c for c in ["game_id", "date", "home_team", "away_team"] if c in df.columns]
    assert df.duplicated(subset=keys).sum() == 0
