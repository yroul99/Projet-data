"""
Tests de qualité : unicité, KPI, résidus
Valide les livrables sérieusement.
"""
import json
import math
import pandas as pd
import pytest
import numpy as np


class TestDataQuality:
    """Suite de tests : Unicité, KPI, Résidus"""

    def test_01_uniqueness_no_duplicates(self, tmp_repo):
        """
        ✅ Unicité : après clean_data.py, aucune clé
        (date, home_team, away_team) n'est dupliquée
        """
        from src.utils.clean_data import clean_2021
        from config import CLEAN_FILE

        clean_2021()
        df = pd.read_csv(CLEAN_FILE)

        keys = ["date", "home_team", "away_team"]
        assert all(k in df.columns for k in keys), f"Colonnes manquantes: {keys}"

        duplicates = df.duplicated(subset=keys, keep=False)
        n_dupes = duplicates.sum()
        assert n_dupes == 0, (
            f"❌ {n_dupes} doublons détectés:\n" f"{df[duplicates][keys].to_string()}"
        )
        print(f"✅ Unicité vérifiée: {len(df)} matches uniques")

    def test_02_summary_json_complete_structure(self, tmp_repo):
        """
        ✅ KPI : save_summary.py produit JSON avec clés attendues:
        - baseline: n, home_diff_mean, home_diff_ci95
        - elo_linear: n, b0, b0_ci95, b0_p, alpha, alpha_ci95, alpha_p
        """
        import importlib
        import config

        from src.utils.clean_data import clean_2021
        from scripts.save_summary import main as save_summary

        importlib.reload(config)
        clean_2021()
        save_summary()

        assert config.SUMMARY_JSON.exists(), "summary.json absent"
        summary = json.loads(config.SUMMARY_JSON.read_text())

        # Structure baseline
        assert "baseline" in summary, "❌ Clé 'baseline' manquante"
        baseline = summary["baseline"]

        for key in ["n", "home_diff_mean", "home_diff_ci95"]:
            assert key in baseline, f"❌ baseline.{key} manquant"

        assert len(baseline["home_diff_ci95"]) == 2, "CI95 doit avoir 2 bornes"
        assert (
            baseline["home_diff_ci95"][0]
            <= baseline["home_diff_mean"]
            <= baseline["home_diff_ci95"][1]
        ), "❌ Moyenne en dehors des IC95%"

        # Structure elo_linear
        assert "elo_linear" in summary, "❌ Clé 'elo_linear' manquante"
        elo = summary["elo_linear"]

        for key in ["n", "b0", "b0_ci95", "b0_p", "alpha", "alpha_ci95", "alpha_p"]:
            assert key in elo, f"❌ elo_linear.{key} manquant"

        assert len(elo["alpha_ci95"]) == 2, "alpha_ci95 doit avoir 2 bornes"

        print(f"✅ Summary KPI complet:")
        print(f"   - Baseline n={baseline['n']}, mean={baseline['home_diff_mean']:.2f}")
        print(f"   - Elo: α={elo['alpha']:.3f}, b₀={elo['b0']:.3f}")

    def test_03_residuals_mean_approximately_zero(self, tmp_repo):
        """
        ✅ Résidus : reconstruction de résiduel
        residual = home_diff - α*elo_delta_pre - b₀
        doit avoir moyenne ≈ 0 (tolérance ±0.5)
        """
        import importlib
        import config

        from src.utils.clean_data import clean_2021
        from scripts.save_summary import main as save_summary

        importlib.reload(config)
        clean_2021()
        save_summary()

        df = pd.read_csv(config.CLEAN_FILE)
        summary = json.loads(config.SUMMARY_JSON.read_text())

        alpha = summary["elo_linear"]["alpha"]
        b0 = summary["elo_linear"]["b0"]

        # Colonnes requises
        required = ["home_diff", "elo_delta_pre"]
        if not all(c in df.columns for c in required):
            pytest.skip(f"Colonnes manquantes: {required}")

        # Nettoyer les NaN
        df_clean = df.dropna(subset=required).copy()
        if len(df_clean) < 2:
            pytest.skip("Pas assez de données valides pour calculer résiduel")

        # Reconstruire résiduel
        df_clean["residual_reconstructed"] = (
            df_clean["home_diff"] - alpha * df_clean["elo_delta_pre"] - b0
        )

        mean_residual = df_clean["residual_reconstructed"].mean()
        std_residual = df_clean["residual_reconstructed"].std()
        tolerance = 0.5

        assert abs(mean_residual) < tolerance, (
            f"❌ Moyenne résiduelle = {mean_residual:.4f}, "
            f"attendu proche de 0 (tolérance ±{tolerance})\n"
            f"   α={alpha:.4f}, b₀={b0:.4f}, σ={std_residual:.4f}"
        )
        print(f"✅ Résidus validés:")
        print(f"   - Mean(residual) = {mean_residual:.4f} (tolérance ±{tolerance})")
        print(f"   - Std(residual) = {std_residual:.4f}")
