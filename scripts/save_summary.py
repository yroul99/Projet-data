# scripts/save_summary.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import CLEAN_FILE

OUT = Path("data/cleaned/summary.json")

def ci95_bootstrap_mean(x: np.ndarray, B: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(B, x.size), replace=True)
    means = boots.mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

def main():
    df = pd.read_csv(CLEAN_FILE)

    # ---- Baseline: moyenne home_diff + IC95 bootstrap
    x = df["home_diff"].dropna().to_numpy()
    lo, hi = ci95_bootstrap_mean(x)
    baseline = {
        "n": int(df.shape[0]),
        "home_diff_mean": float(x.mean()),
        "home_diff_ci95": [lo, hi],
    }

    # ---- Intercept b0 et pente alpha via OLS: home_diff ~ 1 + elo_delta_pre
    if {"home_diff", "elo_delta_pre"}.issubset(df.columns):
        m = df[["home_diff", "elo_delta_pre"]].dropna()
        X = sm.add_constant(m["elo_delta_pre"])
        y = m["home_diff"]
        ols = sm.OLS(y, X).fit(cov_type="HC1")

        params = ols.params
        ci = ols.conf_int()

        out_elo = {
            "n": int(m.shape[0]),
            "b0": float(params["const"]),
            "b0_ci95": [float(ci.loc["const", 0]), float(ci.loc["const", 1])],
            "b0_p": float(ols.pvalues["const"]),
            "alpha": float(params["elo_delta_pre"]),
            "alpha_ci95": [float(ci.loc["elo_delta_pre", 0]), float(ci.loc["elo_delta_pre", 1])],
            "alpha_p": float(ols.pvalues["elo_delta_pre"]),
        }
    else:
        out_elo = {"note": "elo_delta_pre/home_diff manquants dans le CSV."}

    summary = {"baseline": baseline, "elo_linear": out_elo}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OK] summary → {OUT}")

if __name__ == "__main__":
    main()
