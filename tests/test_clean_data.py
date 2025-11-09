import pandas as pd
from src.utils.clean_data import _rest_features

def test_rest_features_b2b_flag():
    df = pd.DataFrame({
        "date": ["2021-10-20","2021-10-21","2021-10-25"],
        "home_team": ["A","A","A"],
        "away_team": ["B","C","D"],
        "season": [2021,2021,2021],
        "home_pts":[1,1,1],"away_pts":[0,0,0]
    })
    out = _rest_features(df)
    assert {"home_b2b","away_b2b","rest_delta"}.issubset(out.columns)
    # le match du 21 est B2B pour A
    assert bool(out.loc[out["date"]=="2021-10-21","home_b2b"].iloc[0]) is True

def test_dataset_has_elevation_columns_after_clean(monkeypatch):
    # smoke test sur le CSV nettoyé
    import os
    assert os.path.exists("data/cleaned/dataset_clean.csv")
    d = pd.read_csv("data/cleaned/dataset_clean.csv")
    assert {"elev_m","home_elev_m","away_elev_m","delta_elev"}.issubset(d.columns)
