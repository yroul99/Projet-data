"""Vérifications rapides pour garder dataset nettoyé et summary alignés."""
import json
import pandas as pd
from config import CLEAN_FILE, SUMMARY_JSON

df = pd.read_csv(CLEAN_FILE)
assert (
    df.duplicated(
        subset=[
            c for c in ["game_id", "date", "home_team", "away_team"] if c in df.columns
        ]
    ).sum()
    == 0
), "Doublons détectés"
assert {"lat", "lon"}.issubset(df.columns) and df[
    ["lat", "lon"]
].notna().all().all(), "lat/lon manquants"
with SUMMARY_JSON.open() as f:
    s = json.load(f)
assert s.get("baseline", {}).get("n") == len(df), "summary.json n != nombre de lignes"

print("[OK] dataset propre, sans doublons, coords complètes, summary aligné.")
