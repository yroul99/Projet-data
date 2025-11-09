# scripts/save_summary.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import CLEAN_FILE

OUT = Path("data/cleaned/summary.json")


def load_clean_df() -> pd.DataFrame:
    """Charge le CSV, crée home_diff si besoin, et déduplique les matchs."""
    p = Path(CLEAN_FILE)
    df = pd.read_csv(p)

    if "home_diff" not in df.columns and {"home_pts", "away_pts"}.issubset(df.columns):
        df["home_diff"] = df["home_pts"] - df["away_pts"]

    keys = [c for c in ["date", "home_team", "away_team"] if c in df.columns]
    if keys:
        df = (
            df.sort_values(keys)
              .drop_duplicates(subset=keys, keep="first")
              .reset_index(drop=True)
        )
    return df


def ci95(mean: float, std: float, n: int) -> list[float]:
    se = std / np.sqrt(n)
    return [float(mean - 1.96 * se), float(mean + 1.96 * se)]


def main() -> None:
    df = load_clean_df()

    # ---- Baseline (home_diff) ----
    y = df["home_diff"].astype(float)
    n = int(y.notna().sum())
    m = float(y.mean())
    sd = float(y.std(ddof=1))
    baseline = {
        "n": n,
        "home_diff_mean": m,
        "home_diff_ci95": ci95(m, sd, n),
    }

    # ---- Régression linéaire : home_diff ~ b0 + alpha * (Elo_home - Elo_away) ----
    if "elo_delta_pre" in df.columns:
        xdelta = df["elo_delta_pre"].astype(float)
        xname = "elo_delta_pre"
    elif {"elo_home_pre", "elo_away_pre"}.issubset(df.columns):
        xdelta = (df["elo_home_pre"] - df["elo_away_pre"]).astype(float)
        xname = "elo_home_pre - elo_away_pre"
    else:
        raise SystemExit("Colonnes Elo absentes pour la régression.")

    X = sm.add_constant(xdelta)
    ols = sm.OLS(y, X, missing="drop").fit()

    ci = ols.conf_int()
    b0 = float(ols.params["const"])
    alpha = float(ols.params[X.columns[1]])
    b0_ci = [float(v) for v in ci.loc["const"].tolist()]
    alpha_ci = [float(v) for v in ci.loc[X.columns[1]].tolist()]

    out = {
        "baseline": baseline,
        "elo_linear": {
            "n": int(ols.nobs),
            "b0": b0,
            "b0_ci95": b0_ci,
            "b0_p": float(ols.pvalues["const"]),
            "alpha": alpha,
            "alpha_ci95": alpha_ci,
            "alpha_p": float(ols.pvalues[X.columns[1]]),
            "x_name": xname,
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"[OK] summary → {OUT}")


if __name__ == "__main__":
    main()
