"""Agrége les statistiques et écrit summary.json pour le dashboard."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm  # requis par les tests pour l'OLS


def ci95_mean(x: pd.Series) -> tuple[float, float]:
    """Calcule un IC95% (approx. normale) pour la moyenne observée."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    hw = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    return (m - hw, m + hw)


def fit_ols_home_vs_elo(df: pd.DataFrame):
    """
    Estime: home_diff = b0 + alpha * elo_delta_pre + eps
    Retourne (b0, b0_ci, b0_p, alpha, alpha_ci, alpha_p)
    """
    cols = {"home_diff", "elo_delta_pre"}
    if not cols.issubset(df.columns):
        # fallback neutre si les colonnes ne sont pas là
        return (np.nan, (np.nan, np.nan), np.nan, np.nan, (np.nan, np.nan), np.nan)

    d = df.dropna(subset=list(cols)).copy()
    if d.empty:
        return (np.nan, (np.nan, np.nan), np.nan, np.nan, (np.nan, np.nan), np.nan)

    X = sm.add_constant(d["elo_delta_pre"].astype(float), has_constant="add")
    y = d["home_diff"].astype(float)
    model = sm.OLS(y, X, missing="drop").fit()

    # params: index ["const", "elo_delta_pre"]
    b0 = float(model.params.get("const", np.nan))
    alpha = float(model.params.get("elo_delta_pre", np.nan))

    ci = model.conf_int()  # DataFrame 2 colonnes [lower, upper]
    b0_ci = tuple(ci.loc["const"].tolist()) if "const" in ci.index else (np.nan, np.nan)
    alpha_ci = (
        tuple(ci.loc["elo_delta_pre"].tolist())
        if "elo_delta_pre" in ci.index
        else (np.nan, np.nan)
    )

    b0_p = float(model.pvalues.get("const", np.nan))
    alpha_p = float(model.pvalues.get("elo_delta_pre", np.nan))

    return (b0, b0_ci, b0_p, alpha, alpha_ci, alpha_p)


def main():
    """Calcule les statistiques depuis CLEAN_FILE et met à jour summary."""
    # Reimport config to respect any monkeypatching (e.g., in tests)
    import importlib
    import config

    importlib.reload(config)

    # On lit le fichier "propre" actuel (honore DATA_DIR si défini par les tests)
    df = pd.read_csv(config.CLEAN_FILE)

    # Baseline sur la marge à domicile brute
    n = int(df["home_diff"].notna().sum()) if "home_diff" in df else 0
    home_diff_mean = (
        float(df["home_diff"].mean()) if "home_diff" in df else float("nan")
    )
    home_diff_ci = (
        ci95_mean(df["home_diff"])
        if "home_diff" in df
        else (float("nan"), float("nan"))
    )

    # Régression OLS: home_diff ~ const + elo_delta_pre
    b0, b0_ci, b0_p, alpha, alpha_ci, alpha_p = fit_ols_home_vs_elo(df)

    out = {
        "baseline": {
            "n": n,
            "home_diff_mean": home_diff_mean,
            "home_diff_ci95": [home_diff_ci[0], home_diff_ci[1]],
        },
        "elo_linear": {
            "n": int(df.dropna(subset=["home_diff"]).shape[0])
            if "home_diff" in df
            else 0,
            "b0": b0,
            "b0_ci95": [b0_ci[0], b0_ci[1]],
            "b0_p": b0_p,
            "alpha": alpha,
            "alpha_ci95": [alpha_ci[0], alpha_ci[1]],
            "alpha_p": alpha_p,
        },
    }

    config.SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.SUMMARY_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] summary → {config.SUMMARY_JSON}")


if __name__ == "__main__":
    main()
