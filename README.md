# NBA Home Advantage Dashboard

Tableau de bord Dash analysant l'avantage du terrain pendant la saison NBA 2021‑22 grâce aux données balldontlie (matchs), Wikidata (arènes) et Open‑Elevation (altitude). Le pipeline nettoie, enrichit et agrège les matchs, calcule une régression ajustée Elo, puis alimente la page unique située dans `src/pages/home.py`.

---

## User Guide

### 1. Pré requis
- Windows / macOS / Linux avec **Python 3.9+**.
- Accès internet si vous souhaitez re-télécharger les données brutes.
- PowerShell (Windows) ou bash pour exécuter les commandes ci-dessous.

### 2. Installation
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Préparation des données
- Le dépôt embarque déjà `data/cleaned/dataset_clean.csv` et `data/cleaned/summary.json`.
- Pour tout recalculer (téléchargement balldontlie + wikidata + nettoyage + Elo + summary) :
```powershell
python -m src.utils.get_data         # téléchargement API / références
python -m src.utils.clean_data       # nettoyage / fusion / features
python -m scripts.save_summary       # calcul des KPI & export summary.json
```
- Placez les fichiers générés dans `data/cleaned/` (chemins configurés via `config.py`).

### 4. Lancer le dashboard
```powershell
python main.py
```
Le serveur Dash écoute sur `http://127.0.0.1:8050`. Pour déployer sur une autre machine, répétez l’installation, assurez-vous que les fichiers `data/cleaned/*.csv` et `summary.json` sont disponibles, puis exposez le port 8050 (ou définissez `PORT` dans les variables d’environnement).

### 5. Tests et qualité
```powershell
python -m pytest
ruff check .
black --check .
```
`pre-commit install` permet de lancer automatiquement Black + Ruff sur chaque commit.

---

## Data

| Source | Format | Emplacement | Description |
| --- | --- | --- | --- |
| balldontlie (API) | JSON → `data/raw/balldontlie_games_2021.json` | Matchs de saison régulière 2021‑22 avec scores et identifiants équipes |
| Wikidata | CSV/JSON → `data/raw/wikidata_arenas.*` | Métadonnées arènes : nom, lat/lon, capacité |
| Open‑Elevation | JSON cache → `data/raw/open_elevation_cache.json` | Altitude des arènes (mètres) |
| Dataset nettoyé | CSV → `data/cleaned/dataset_clean.csv` | Table consolidée (home/away, Elo pré-match, delta altitude, résidu) |
| Résumé KPI | JSON → `data/cleaned/summary.json` | Statistiques agrégées (home_diff moyen, intercept résiduel, alpha, n) |

Le nettoyage applique :
1. Dédoublonnage sur `(date, home_team, away_team)`.
2. Calcul `home_diff`, résidu Elo (`residual_margin`), repos, delta altitude.
3. Jointure sur l’inventaire `arenas_unique.csv` pour récupérer capacité et altitude.
4. Export final dans `data/cleaned/` + `summary.json` utilisé par l’interface.

---

## Developer Guide

### Architecture générale

```mermaid
flowchart TD
	main[main.py] -->|Initialise Dash app| layout[src/pages/home.py]
	layout -->|register_callbacks()| callbacks[(Callbacks)]
	callbacks --> dataLoader[load_df / summary_block]
	dataLoader --> files[data/cleaned/dataset_clean.csv\nsummary.json]
	callbacks --> utils[src/utils/*]
	scripts[scripts/*.py] --> files
	tests[tests/*.py] --> main
```

- `main.py` : crée l’application Dash, importe `src/pages/home.py` et enregistre les callbacks.
- `src/pages/home.py` : layout complet + callbacks (`_update_global`, `_update_duel`).
- `src/utils/clean_data.py`, `get_data.py`, `elo.py` : pipeline acquisition / features / Elo.
- `scripts/` : scripts d’analyse ponctuels (`save_summary.py`, `analyze_*`).
- `tests/` : tests unitaires et de pipeline.

### Ajouter une nouvelle page
1. Créer un module `src/pages/<nouvelle_page>.py` exposant `layout` et `register_callbacks(app)`.
2. Dans `main.py`, importer la page et ajouter une route Dash (ou un onglet `dcc.Location`).
3. Réutiliser `load_df()` pour bénéficier du cache et de l’enrichissement altitude.

### Ajouter un graphique à la page existante
1. Ajouter le composant Dash (ex. `dcc.Graph(id="nouveau-graph")`) dans `layout`.
2. Étendre la callback `_update_global` ou créer une nouvelle callback dédiée.
3. Utiliser les helpers `_build_arenas_df` ou `load_df()` pour accéder aux données.
4. Documenter le graphique (texte explicatif) afin de conserver l’approche pédagogique.

---

## Rapport d'analyse

- **Cartographie arènes** : les salles en altitude (Denver, Utah) cumulent souvent capacité importante et contrainte géographique, accentuant la fatigue des équipes visiteuses.
- **Histogrammes** : la marge brute moyenne tourne autour de +1.67 pts pour l’équipe à domicile. L’ajustement Elo (résiduel) recentre la distribution proche de 0, montrant qu’une large part de l’avantage provient des différences de niveau.
- **Altitude vs résiduel** : tendance légèrement positive → même après correction Elo, les matchs disputés en altitude conservent un bonus résiduel.
- **Classement + marges** : les leaders (Suns, Grizzlies, Warriors, Heat, Celtics) dominent autant chez eux qu’au global. Certaines exceptions (Jazz, Nuggets) présentent une marge domicile supérieure à leur rang global, signe d’un public/altitude particulièrement favorable.
- **Duel interactif** : impose que l’équipe 1 soit à domicile et liste chaque match, ce qui aide à isoler les confrontations spécifiques.

Ces observations suggèrent que l’avantage du terrain résulte d’un cocktail : qualité intrinsèque, ferveur locale (capacité), logistique (repos / altitude).

---

## Copyright

Je déclare sur l’honneur que le code fourni dans ce dépôt a été produit par moi-même, à l’exception des éléments listés ci-dessous :

- **Plotly / Dash snippets** : la structure générale des callbacks suit la documentation officielle Dash/Plotly (https://dash.plotly.com/). Les signatures `@app.callback([...], [...])` et l’utilisation de `dcc.Graph` sont conformes aux exemples fournis par Plotly.

Toute ligne non déclarée ci-dessus est réputée être produite par l’auteur du projet. L’absence de mention supplémentaire signifie qu’aucune autre portion n’a été copiée.

---

## Licence & contact

- Licence : MIT (mettre à jour selon vos besoins).
- Auteur : Yoan Roul (alias `yroul99`).
- Pour toute question : ouvrir une issue GitHub ou envoyer un e-mail.
