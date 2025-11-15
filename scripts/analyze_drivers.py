"""Explore quels facteurs expliquent les résidus via plusieurs modèles OLS."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from config import CLEAN_FILE


def z(s):
    """Normalise en z-score en ignorant les valeurs non numériques."""
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std(ddof=0)


df = pd.read_csv(CLEAN_FILE)

# ----- features -----
df["log_capacity"] = np.log(
    df["home_capacity"].where(df["home_capacity"].notna(), df["capacity"])
)
df["log_capacity_z"] = z(df["log_capacity"])
df["rest_delta_z"] = z(df["rest_delta"])
df["elev_m_z"] = z(df["elev_m"])  # altitude domicile (pour comparaison)
df["delta_elev_z"] = z(df["delta_elev"])  # clé: delta altitude domicile – extérieur

# ----- M1 : sans effets fixes, delta_elev -----
f1 = "residual_margin ~ home_b2b + away_b2b + rest_delta_z + log_capacity_z + delta_elev_z"
m1 = smf.ols(f1, data=df).fit(cov_type="HC1")
print("\n[M1] Sans effets fixes (SE robustes) — delta_elev_z :")
print(m1.summary())

# ----- M2 : effets fixes home + away (two-way FE), delta_elev -----
teams = "C(home_team) + C(away_team)"
f2 = (
    "residual_margin ~ home_b2b + away_b2b + rest_delta_z + "
    "log_capacity_z + delta_elev_z + C(home_team) + C(away_team)"
)  # noqa: E501
m2 = smf.ols(f2, data=df).fit(cov_type="HC1")
print("\n[M2] Effets fixes (home_team & away_team) — " "SE robustes — delta_elev_z :")
print(m2.summary())

# ----- (option) comparaison avec elev_m_z (ancienne spec) -----
f2_elev = (
    "residual_margin ~ home_b2b + away_b2b + rest_delta_z + "
    "log_capacity_z + elev_m_z + C(home_team) + C(away_team)"
)
m2e = smf.ols(f2_elev, data=df).fit(cov_type="HC1")
print("\n[Comparaison] Effets fixes avec elev_m_z (altitude domicile seule) :")
print(m2e.summary().tables[1])

# ----- groupes par delta_elev (binning en mètres) -----
bins = [-1e9, -800, -300, 300, 800, 1e9]
labels = ["≪-800m", "-800–-300", "-300–300", "300–800", "≫800m"]
df["delta_bin"] = pd.cut(
    df["delta_elev"], bins=bins, labels=labels, include_lowest=True
)
print("\n[Groupes Δ altitude] mean(residual_margin) par bin:")
print(df.groupby("delta_bin")["residual_margin"].agg(["count", "mean", "std"]).round(2))

# ----- petite vérif Denver -----
print(
    "\n[Denver focus] mean(residual_margin) par bin adversaire "
    "(Δ altitude vus depuis Denver domicile):"
)
den = df[df["home_team"] == "Denver Nuggets"]
print(
    den.groupby("delta_bin")["residual_margin"].agg(["count", "mean", "std"]).round(2)
)
