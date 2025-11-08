from pathlib import Path
import numpy as np
import pandas as pd

# Statsmodels uniquement (évite SciPy si pas installé)
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned" / "dataset_clean.csv"

df = pd.read_csv(DATA)

# 1) Edge BRUT = moyenne(home_diff)
home_diff = df["home_diff"].dropna()
n = len(home_diff)
mean_brut = home_diff.mean()
std_brut = home_diff.std(ddof=1)
se_brut = std_brut / np.sqrt(n)

# IC95% approx (normal) — suffisant vu n≈1200
ci_low = mean_brut - 1.96 * se_brut
ci_high = mean_brut + 1.96 * se_brut

print(f"[BRUT] mean(home_diff) = {mean_brut:.2f}  (IC95% ≈ [{ci_low:.2f}, {ci_high:.2f}])  n={n}")

# 2) Edge AJUSTÉ = intercept d’une régression home_diff ~ const + elo_delta_pre
X = sm.add_constant(df["elo_delta_pre"].fillna(0.0))
y = df["home_diff"]
model = sm.OLS(y, X, missing="drop").fit()
b0 = model.params.get("const", float("nan"))
ci_b0 = model.conf_int().loc["const"].tolist()
p_b0 = model.pvalues.get("const", float("nan"))

print("\n[AJUSTÉ Elo]")
print(f"b0 (intercept) = {b0:.3f}   IC95% = [{ci_b0[0]:.3f}, {ci_b0[1]:.3f}]   p={p_b0:.3g}")
print("(Si l’IC de b0 n’inclut pas 0 → avantage du terrain persistant, même à Elo neutre.)")
