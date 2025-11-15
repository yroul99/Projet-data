"""Implémentation Elo minimale avec ajustement MOV pour la NBA."""
from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple

START_ELO = 1500.0
DEFAULT_K = 20


def _expected_from_delta(delta: float) -> float:
    """Probabilité de victoire sur terrain neutre selon le delta Elo."""
    return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))


def _mov_multiplier(mov: float, delta: float) -> float:
    """Multiplicateur marge de victoire proche de la recette 538."""
    return ((abs(mov) + 3.0) ** 0.8) / (7.5 + 0.006 * abs(delta))


def run_elo(
    df_games: pd.DataFrame,
    k: int = DEFAULT_K,
    use_mov: bool = True,
    start_rating: float = START_ELO,
) -> pd.DataFrame:
    """
    df_games doit contenir: ['date','home_team','away_team','home_pts','away_pts']
    Ajoute: ['elo_home_pre','elo_away_pre','elo_delta_pre','elo_exp_home_win']
    """
    df = df_games.sort_values("date").copy()
    ratings: Dict[str, float] = {}
    elo_home_pre, elo_away_pre, elo_delta_pre, elo_exp = [], [], [], []

    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        rh = ratings.get(h, start_rating)
        ra = ratings.get(a, start_rating)

        delta = rh - ra
        p_home = _expected_from_delta(delta)

        elo_home_pre.append(rh)
        elo_away_pre.append(ra)
        elo_delta_pre.append(delta)
        elo_exp.append(p_home)

        s_home = 1.0 if row["home_pts"] > row["away_pts"] else 0.0
        mult = (
            _mov_multiplier(row["home_pts"] - row["away_pts"], delta)
            if use_mov
            else 1.0
        )
        change = k * mult * (s_home - p_home)
        ratings[h] = rh + change
        ratings[a] = ra - change

    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    df["elo_delta_pre"] = elo_delta_pre
    df["elo_exp_home_win"] = elo_exp
    return df


def fit_expected_margin_and_residual(
    df_with_elo: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Apprend alpha et intercept b0 tels que:
        expected_margin_neutral = b0 + alpha * elo_delta_pre
        residual_margin = (home_pts - away_pts) - expected_margin_neutral
    Retourne (df, alpha, b0)
    """
    df = df_with_elo.copy()
    x = df["elo_delta_pre"]
    y = df["home_pts"] - df["away_pts"]
    xm, ym = x.mean(), y.mean()
    var = ((x - xm) ** 2).sum()
    cov = ((x - xm) * (y - ym)).sum()
    alpha = (cov / var) if var != 0 else 0.0
    b0 = ym - alpha * xm

    df["expected_margin_neutral"] = (b0 + alpha * df["elo_delta_pre"]).round(2)
    df["residual_margin"] = (y - df["expected_margin_neutral"]).round(2)
    return df, alpha, b0
