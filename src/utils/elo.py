# src/utils/elo.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple

START_ELO = 1500.0
DEFAULT_K = 20

def _expected_from_delta(delta: float) -> float:
    """Proba de victoire du home à terrain neutre (logistique base 10, SANS bonus domicile)."""
    return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))

def _mov_multiplier(mov: float, delta: float) -> float:
    """
    Multiplicateur marge de victoire (style 538).
    mov: margin of victory (valeur absolue)
    delta: écart Elo pré-match (home - away), valeur absolue
    """
    return ((mov + 3.0) ** 0.8) / (7.5 + 0.006 * abs(delta))

def run_elo(df_games: pd.DataFrame, k: int = DEFAULT_K, use_mov: bool = True,
            start_rating: float = START_ELO) -> pd.DataFrame:
    """
    Entrée df trié par date avec colonnes:
      ['date','home_team','away_team','home_pts','away_pts']
    Retourne df avec colonnes ajoutées:
      ['elo_home_pre','elo_away_pre','elo_delta_pre','elo_exp_home_win']
    et met à jour itérativement les ratings par équipe.
    """
    df = df_games.sort_values("date").copy()
    ratings: Dict[str, float] = {}
    elo_home_pre, elo_away_pre, elo_delta_pre, elo_exp = [], [], [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh = ratings.get(h, start_rating)
        ra = ratings.get(a, start_rating)

        delta = rh - ra  # terrain neutre
        p_home = _expected_from_delta(delta)

        # stocke les valeurs pré-match
        elo_home_pre.append(rh)
        elo_away_pre.append(ra)
        elo_delta_pre.append(delta)
        elo_exp.append(p_home)

        # résultat (1 si home gagne, 0 sinon)
        s_home = 1.0 if row["home_pts"] > row["away_pts"] else 0.0

        # facteur de mise à jour
        mult = 1.0
        if use_mov:
            mov = abs(row["home_pts"] - row["away_pts"])
            mult = _mov_multiplier(mov, delta)

        change = k * mult * (s_home - p_home)
        ratings[h] = rh + change
        ratings[a] = ra - change

    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    df["elo_delta_pre"] = elo_delta_pre
    df["elo_exp_home_win"] = elo_exp
    return df

def fit_expected_margin_and_residual(df_with_elo: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Apprend la pente alpha qui relie la marge au delta Elo (attendu neutre):
        expected_margin_neutral = alpha * elo_delta_pre
        residual_margin = (home_pts - away_pts) - expected_margin_neutral
    """
    df = df_with_elo.copy()
    x = df["elo_delta_pre"]
    y = df["home_pts"] - df["away_pts"]
    xm, ym = x.mean(), y.mean()
    var = ((x - xm) ** 2).sum()
    cov = ((x - xm) * (y - ym)).sum()
    alpha = (cov / var) if var != 0 else 0.0

    df["expected_margin_neutral"] = (alpha * df["elo_delta_pre"]).round(2)
    df["residual_margin"] = (y - df["expected_margin_neutral"]).round(2)
    return df, alpha
