"""Mise en page et callbacks Dash du tableau de bord avantage domicile."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output

from config import CLEAN_FILE, CLEANED_DIR, RAW_ELEV, SEASON, SUMMARY_JSON

# ---------------------------------------------------------------------
# Constantes / libellés / styles
# ---------------------------------------------------------------------
SUMMARY_PATH = SUMMARY_JSON
ARENAS_PATH = CLEANED_DIR / "arenas_unique.csv"
ELEV_CACHE_PATH = RAW_ELEV

LABELS = {
    "home_diff": "Marge à domicile (pts)",
    "residual_margin": "Marge résiduelle (pts, ajustée Elo)",
}

KPI_WRAPPER_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(4, minmax(160px, 1fr))",
    "gap": "12px",
}
KPI_CARD_STYLE = {
    "background": "white",
    "border": "1px solid #e5e7eb",
    "borderRadius": "10px",
    "padding": "10px 12px",
}
KPI_LABEL_STYLE = {"fontSize": "12px", "color": "#6b7280"}
KPI_VALUE_STYLE = {"fontSize": "20px", "fontWeight": 600, "marginTop": "2px"}
KPI_NOTE_STYLE = {"fontSize": "12px", "color": "#6b7280", "marginTop": "4px"}


# ---------------------------------------------------------------------
# Chargement & cache
# ---------------------------------------------------------------------
def _read_df_from_disk() -> pd.DataFrame:
    """Charge le dataset nettoyé (saison complète) depuis le disque."""
    p = Path(CLEAN_FILE)
    df = pd.read_csv(p)

    # déduplication basique
    key = [c for c in ("date", "home_team", "away_team") if c in df.columns]
    if key:
        df = (
            df.sort_values(key)
            .drop_duplicates(subset=key, keep="first")
            .reset_index(drop=True)
        )

    # home_diff si absent
    if "home_diff" not in df.columns and {"home_pts", "away_pts"}.issubset(df.columns):
        df["home_diff"] = df["home_pts"] - df["away_pts"]

    return df


@lru_cache(maxsize=1)
def load_df_cached() -> pd.DataFrame:
    """Charge une fois le CSV en mémoire (cache)."""
    return _read_df_from_disk()


def load_df() -> pd.DataFrame:
    """Retourne une copie des données pour les callbacks."""
    df = load_df_cached().copy()
    if "home_diff" not in df and {"home_pts", "away_pts"}.issubset(df):
        df["home_diff"] = df["home_pts"] - df["away_pts"]
    # dédup stricte via game_id ou (date, home_team, away_team)
    for cols in [["game_id"], ["date", "home_team", "away_team"]]:
        if all(c in df.columns for c in cols):
            df = df.sort_values(cols).drop_duplicates(subset=cols, keep="first")
            break

    df = _enrich_with_elevation(df)
    return df


# ---------------------------------------------------------------------
# Résumé KPI
# ---------------------------------------------------------------------
def read_summary() -> Dict[str, Any]:
    """Lit summary.json produit par scripts/save_summary.py."""
    if SUMMARY_PATH.exists():
        try:
            return json.loads(SUMMARY_PATH.read_text())
        except Exception:
            pass
    return {}


def _fmt_ci(ci) -> str:
    """Formate un intervalle de confiance [lo, hi]."""
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci and len(ci) == 2 else "—"


def summary_block() -> html.Div:
    """Bloc de résumé (KPI) affiché en haut de la page."""
    s = read_summary()
    if not s:
        return html.Div()

    base = s.get("baseline", {})
    elo = s.get("elo_linear", {})

    def kpi(title: str, value: str, note: str = "") -> html.Div:
        return html.Div(
            [
                html.Div(title, style=KPI_LABEL_STYLE),
                html.Div(value, style=KPI_VALUE_STYLE),
                html.Div(note, style=KPI_NOTE_STYLE),
            ],
            style=KPI_CARD_STYLE,
        )

    return html.Div(
        [
            html.H4(
                f"Résumé ({SEASON}-{SEASON + 1})",
                style={"marginBottom": "8px"},
            ),
            html.Div(
                [
                    kpi("n", f"{int(base.get('n', 0))}"),
                    kpi(
                        "home_diff moyen",
                        f"{base.get('home_diff_mean', 0):.2f}",
                        f"IC95% {_fmt_ci(base.get('home_diff_ci95'))}",
                    ),
                    kpi(
                        "b₀ (résiduel Elo)",
                        f"{elo.get('b0', 0):.2f}",
                        (
                            f"IC95% {_fmt_ci(elo.get('b0_ci95'))} • "
                            f"p={elo.get('b0_p', 0):.2e}"
                        ),
                    ),
                    kpi(
                        "α (Elo decay)",
                        f"{elo.get('alpha', 0):.3f}",
                        f"IC95% {_fmt_ci(elo.get('alpha_ci95'))}",
                    ),
                ],
                style=KPI_WRAPPER_STYLE,
            ),
        ],
        style={
            "background": "#f8f9fb",
            "border": "1px solid #e5e7eb",
            "borderRadius": "10px",
            "padding": "12px 14px",
            "marginBottom": "16px",
        },
    )


# ---------------------------------------------------------------------
# Bornes RangeSlider (Δ altitude)
# ---------------------------------------------------------------------
def _slider_bounds(df: pd.DataFrame) -> Tuple[int, int]:
    """Calcule des bornes arrondies pour le RangeSlider Δ altitude."""
    if "delta_elev" in df.columns and df["delta_elev"].notna().any():
        vmin = int(math.floor(df["delta_elev"].min() / 100) * 100)
        vmax = int(math.ceil(df["delta_elev"].max() / 100) * 100)
        if vmin == vmax:
            vmin, vmax = vmin - 1000, vmax + 1000
        return vmin, vmax
    return -1700, 1700


# ---------------------------------------------------------------------
# Référentiels arènes / altitude
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_elevation_cache() -> dict:
    """Charge le cache altitude (clé = coordonnées arrondies)."""
    if not ELEV_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(ELEV_CACHE_PATH.read_text())
    except Exception:
        return {}
    return {
        (round(entry.get("latitude"), 6), round(entry.get("longitude"), 6)): entry.get(
            "elevation"
        )
        for entry in data
        if entry.get("latitude") is not None and entry.get("longitude") is not None
    }


@lru_cache(maxsize=1)
def _load_arenas_reference() -> pd.DataFrame:
    """Charge l'inventaire des arènes en complétant les altitudes."""
    if not ARENAS_PATH.exists():
        return pd.DataFrame()
    arenas = pd.read_csv(ARENAS_PATH).copy()

    elev_cache = _load_elevation_cache()
    if "elev_m" not in arenas.columns:
        arenas["elev_m"] = arenas.apply(
            lambda row: elev_cache.get(
                (round(row.get("lat"), 6), round(row.get("lon"), 6))
            ),
            axis=1,
        )
    else:
        missing = arenas["elev_m"].isna()
        if missing.any():
            arenas.loc[missing, "elev_m"] = arenas.loc[missing].apply(
                lambda row: elev_cache.get(
                    (round(row.get("lat"), 6), round(row.get("lon"), 6))
                ),
                axis=1,
            )

    if "capacity" not in arenas.columns:
        arenas["capacity"] = 18000

    return arenas


def _enrich_with_elevation(df: pd.DataFrame) -> pd.DataFrame:
    """Injecte home/away elevation + delta si dispo via arenas_unique."""
    arenas = _load_arenas_reference()
    if arenas.empty or "home_team" not in df.columns or "away_team" not in df.columns:
        return df

    elev_map = {
        team: float(val) if pd.notna(val) else np.nan
        for team, val in arenas.set_index("home_team")["elev_m"].items()
    }

    if "home_elev_m" not in df.columns:
        df["home_elev_m"] = np.nan
    df["home_elev_m"] = pd.to_numeric(df["home_elev_m"], errors="coerce")
    if "away_elev_m" not in df.columns:
        df["away_elev_m"] = np.nan
    df["away_elev_m"] = pd.to_numeric(df["away_elev_m"], errors="coerce")

    home_missing = df["home_elev_m"].isna()
    home_fill = df.loc[home_missing, "home_team"].map(elev_map)
    home_fill = pd.to_numeric(home_fill, errors="coerce")
    df.loc[home_missing, "home_elev_m"] = home_fill

    away_missing = df["away_elev_m"].isna()
    away_fill = df.loc[away_missing, "away_team"].map(elev_map)
    away_fill = pd.to_numeric(away_fill, errors="coerce")
    df.loc[away_missing, "away_elev_m"] = away_fill

    df["elev_m"] = df["home_elev_m"]

    delta = df.get("delta_elev")
    if delta is None:
        df["delta_elev"] = df["home_elev_m"] - df["away_elev_m"]
    else:
        df["delta_elev"] = pd.to_numeric(df["delta_elev"], errors="coerce")
        mask = df["delta_elev"].isna()
        df.loc[mask, "delta_elev"] = (
            df.loc[mask, "home_elev_m"] - df.loc[mask, "away_elev_m"]
        )

    return df


_DF0 = load_df()
EMIN, EMAX = _slider_bounds(_DF0)


# ---------------------------------------------------------------------
# Helpers pour la carte globale
# ---------------------------------------------------------------------
def _build_arenas_df(df: pd.DataFrame) -> pd.DataFrame:
    """Construit un DataFrame d'arènes uniques par franchise."""
    arenas = _load_arenas_reference()
    if not arenas.empty:
        return arenas

    required_min = {"home_team", "arena", "lat", "lon", "elev_m"}
    if not required_min.issubset(df.columns):
        return pd.DataFrame(columns=list(required_min))

    arenas = (
        df[list(required_min)]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates(subset=["home_team"])
    )
    arenas["capacity"] = 18000
    return arenas


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
layout = html.Div(
    [
        html.H1("Étude de l'avantage du terrain de la saison NBA 2021-22"),
        html.P(
            "Analyse conjointe des données balldontlie (matchs), Wikidata (arènes) et "
            "Open-Elevation afin de comprendre comment la géographie et la capacité des salles "
            "influencent les performances à domicile des équipes NBA pour la saison 2021-22.",
            style={"color": "#4b5563", "margin": "6px 0 12px"},
        ),
        dcc.Markdown(
            """
**Objectifs**

- Cartographier les salles NBA en confrontant altitude et capacité.
- Quantifier la distribution des marges à domicile (brute et ajustée Elo).
- Relier l'altitude au résiduel Elo et comparer rapidement deux franchises.
""",
            style={"fontSize": "13px", "color": "#4b5563", "marginBottom": "16px"},
        ),
        dcc.RangeSlider(
            id="delta-elev",
            min=EMIN,
            max=EMAX,
            step=50,
            value=[EMIN, EMAX],
            marks={v: f"{v} m" for v in range(EMIN, EMAX + 1, 500)},
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        dcc.Markdown(
            "Filtre facultatif par **Δ altitude (m)** : altitude domicile − altitude extérieur.",
            style={"fontSize": "12px", "color": "#6b7280", "margin": "8px 0 18px"},
        ),
        html.H3("Cartographie des arènes NBA"),
        html.P(
            "Chaque point représente l'arène principale d'une franchise. Nous calculons un indice = 0,5 × "
            "(altitude normalisée) + 0,5 × (capacité normalisée) afin de résumer en une seule échelle "
            "la contrainte géographique (air plus fin, voyages) et l'effet de foule (volume de supporters). "
            "Les points jaunes signalent les salles à la fois élevées et vastes; les points bleus indiquent des arènes "
            "proches du niveau de la mer ou plus modestes.",
            style={"fontSize": "13px", "color": "#4b5563", "marginBottom": "8px"},
        ),
        dcc.Loading(dcc.Graph(id="map"), type="dot"),
        html.H3("Résumé saison NBA 2021-22"),
        html.H3("Classement 2021-22 (victoires)"),
        html.Div(id="wins-ranking"),
        html.H3("Panorama des marges et classement"),
        html.P(
            "Ce graphique place les 30 franchises sur l'axe horizontal et leur marge moyenne"
            " à domicile (home_pts − away_pts) sur l'axe vertical. Il permet d'identifier en un coup"
            " d'œil quelles équipes dominent systématiquement dans leur salle et lesquelles sont"
            " plus vulnérables malgré l'avantage du terrain.",
            style={"fontSize": "12px", "color": "#374151", "margin": "10px 0 6px"},
        ),
        dcc.Loading(dcc.Graph(id="home-team-bar"), type="dot"),
        html.P(
            "On associe chaque franchise à son volume total de victoires (domicile + extérieur) pour"
            " estimer un classement empirique sur la saison 2021-22. La position horizontale indique"
            " le rang (1 = plus grand nombre de victoires), tandis que l'axe vertical conserve la marge"
            " moyenne à domicile. Couleur et taille reflètent le nombre exact de victoires afin de voir"
            " si une équipe dominante convertit aussi son avantage terrain.",
            style={"fontSize": "12px", "color": "#374151", "margin": "10px 0 6px"},
        ),
        dcc.Loading(dcc.Graph(id="home-margin-rank"), type="dot"),
        html.P(
            "Constat : les deux graphes de marges montrent surtout que les équipes de tête restent performantes partout;"
            " seules quelques exceptions (Utah, Denver) affichent une marge domicile très élevée tout en étant légèrement"
            " en retrait au classement global, signe d'un public et d'un environnement particulièrement difficiles à gérer.",
            style={"fontSize": "12px", "color": "#374151", "margin": "6px 0 10px"},
        ),
        html.P(
            "Synthèse numérique des indicateurs clés issus du pipeline de nettoyage (marge brute, résiduel Elo, n)",
            style={"fontSize": "13px", "color": "#4b5563", "margin": "12px 0 4px"},
        ),
        summary_block(),
        html.P(
            "Le modèle Elo attribue une cote à chaque équipe. Après chaque match, la cote se met à jour "
            "selon la formule Eloₙ₊₁ = Eloₙ + K × (résultat observé − probabilité prévue). Nous l'utilisons "
            "parce qu'il capture la forme des équipes et permet de neutraliser les forces relatives dans les métriques ci-dessous.",
            style={"fontSize": "12px", "color": "#4b5563", "margin": "8px 0 6px"},
        ),
        html.Ul(
            [
                html.Li(
                    "n : nombre de matchs retenus après nettoyage. Chaque ligne est deduplée sur (date, home_team, away_team).",
                    style={"marginBottom": "4px"},
                ),
                html.Li(
                    "home_diff moyen : moyenne(home_pts − away_pts). L'estimation est accompagnée d'un intervalle 95 % = moyenne ± 1.96 × σ/√n;"
                    " ainsi la valeur actuelle (~1.67 pts) signifie qu'une équipe à domicile marque en moyenne 1,7 point de plus que son adversaire.",
                    style={"marginBottom": "4px"},
                ),
                html.Li(
                    "b₀ (résiduel Elo) : intercept du modèle home_diff = b₀ + α × elo_delta_pre. Il mesure la marge prévue lorsque deux"
                    " équipes ont le même Elo (elo_delta_pre = 0) ; la valeur affichée (~1.69 pts) indique qu'en neutralisant Elo,"
                    " il reste un bonus terrain équivalent à presque deux points.",
                    style={"marginBottom": "4px"},
                ),
                html.Li(
                    "α (Elo decay) : pente reliant l'écart de rating Elo à la marge attendue. Une α élevée signifie qu'une différence de Elo"
                    " se traduit fortement en points ; ici α≈0.033 signifie qu'un écart de 100 points Elo se transforme en ~3,3 points"
                    " d'écart à domicile.",
                ),
            ],
            style={"fontSize": "12px", "color": "#4b5563", "margin": "0 0 16px 18px"},
        ),
        html.H3("Histogrammes des marges"),
        html.P(
            "Le premier histogramme illustre la marge brute (home_pts − away_pts). Le second calcule le "
            "résiduel après avoir retiré la contribution attendue α × elo_delta_pre. Comparer les deux "
            "permet de voir si l'avantage domicile subsiste une fois neutralisées les forces relatives.",
            style={"fontSize": "13px", "color": "#4b5563", "marginBottom": "12px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Marge brute à domicile"),
                        html.P(
                            "Différence de score à domicile sans correction Elo.",
                            style={
                                "fontSize": "12px",
                                "color": "#6b7280",
                                "margin": "0 0 6px",
                            },
                        ),
                        dcc.Loading(dcc.Graph(id="hist-home"), type="dot"),
                        html.P(
                            id="hist-home-note",
                            style={
                                "fontSize": "12px",
                                "color": "#374151",
                                "marginTop": "6px",
                            },
                        ),
                    ],
                    style={
                        "background": "white",
                        "padding": "8px",
                        "borderRadius": "8px",
                    },
                ),
                html.Div(
                    [
                        html.H4("Marge résiduelle (ajustée Elo)"),
                        html.P(
                            "On retire l'effet des forces relatives via Elo pour isoler le véritable bonus terrain.",
                            style={
                                "fontSize": "12px",
                                "color": "#6b7280",
                                "margin": "0 0 6px",
                            },
                        ),
                        dcc.Loading(dcc.Graph(id="hist-resid"), type="dot"),
                        html.P(
                            id="hist-resid-note",
                            style={
                                "fontSize": "12px",
                                "color": "#374151",
                                "marginTop": "6px",
                            },
                        ),
                    ],
                    style={
                        "background": "white",
                        "padding": "8px",
                        "borderRadius": "8px",
                    },
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
        ),
        html.P(
            "Constat : la distribution brute reste centrée autour de +1 à +2 points, tandis que la version résiduelle "
            "se resserre proche de 0. Cela montre que l'ajustement Elo explique une grande partie de l'écart, "
            "mais qu'un léger avantage structurel persiste.",
            style={"fontSize": "12px", "color": "#374151", "margin": "8px 0 16px"},
        ),
        html.H3("Altitude vs résiduel"),
        html.P(
            "Chaque point représente un match. Se déplacer vers la droite signifie que l'arène se situe plus haut en altitude; "
            "aller vers la gauche correspond à des salles proches du niveau de la mer. L'axe vertical mesure la marge résiduelle : "
            "une valeur positive indique que l'équipe à domicile a fait mieux qu'attendu selon Elo, tandis qu'une valeur négative "
            "montre qu'elle a sous-performé par rapport à la prédiction.",
            style={"fontSize": "13px", "color": "#4b5563", "margin": "12px 0 6px"},
        ),
        dcc.Loading(dcc.Graph(id="scatter"), type="dot"),
        html.P(
            "La tendance (ligne OLS) permet de voir si le résiduel augmente avec l'altitude : une pente positive "
            "signifie que les équipes haut perchées conservent un bonus même après correction Elo. On observe ici "
            "une pente légèrement ascendante, ce qui confirme que plus l'altitude grimpe, plus l'écart ajusté reste "
            "légèrement favorable aux équipes locales.",
            style={"fontSize": "12px", "color": "#374151", "margin": "6px 0 16px"},
        ),
        html.H3("Comparer deux équipes"),
        html.P(
            "Choisissez explicitement l'équipe à domicile (menu gauche) puis l'adversaire en déplacement (menu droit) "
            "pour afficher la marge obtenue match par match.",
            style={"fontSize": "13px", "color": "#4b5563", "margin": "12px 0 6px"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="team-1",
                    placeholder="Choisir l'équipe 1...",
                    options=[],
                    style={"width": "260px"},
                ),
                dcc.Dropdown(
                    id="team-2",
                    placeholder="Choisir l'équipe 2...",
                    options=[],
                    style={"width": "260px", "marginLeft": "10px"},
                ),
            ],
            style={"display": "flex", "margin": "10px 0"},
        ),
        html.P(
            "Le résumé liste chaque match disputé dans cette configuration (domicile vs extérieur) et indique la marge "
            "associée. Si la marge est positive, l'équipe à domicile a dominé cette rencontre; une marge négative signifie "
            "qu'elle a été battue malgré l'avantage du terrain. La moyenne affichée sous la liste synthétise ces performances.",
            style={"fontSize": "12px", "color": "#374151", "marginBottom": "8px"},
        ),
        html.Div(id="duel-summary"),
        dcc.Graph(id="duel-bar"),
        html.H3("Synthèse globale"),
        html.P(
            "La carte révèle que certaines salles combinent altitude élevée et grande capacité, offrant à la fois une atmosphère pressurisante et un soutien massif."
            " Les histogrammes confirment qu'un avantage moyen d'environ +1,5 point subsiste même après correction Elo."
            " Le bloc classement + marges montre que les équipes de tête (Phoenix, Memphis, Golden State) dominent aussi à domicile, tandis que des spécialistes comme Utah (7e) ou Denver (7-8e) affichent une marge maison élevée mais légèrement en retrait au classement global."
            " À l'inverse, Miami ou Boston performent à la fois en marge et au classement, ce qui laisse penser que leur avantage terrain reflète surtout leur niveau d'élite."
            " Le scatter altitude/résiduel indique enfin qu'une contrainte géographique subsiste. Pris ensemble, ces résultats suggèrent que l'avantage du terrain naît d'un cocktail entre qualité intrinsèque, ferveur locale et conditions de voyage.",
            style={"fontSize": "13px", "color": "#1f2937", "margin": "18px 0 0"},
        ),
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "16px"},
)


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------
def register_callbacks(app) -> None:
    """Enregistre les callbacks Dash pour la page d'accueil."""

    # ----- Graphiques globaux -----
    @app.callback(
        Output("map", "figure"),
        Output("hist-home", "figure"),
        Output("hist-resid", "figure"),
        Output("scatter", "figure"),
        Output("home-team-bar", "figure"),
        Output("wins-ranking", "children"),
        Output("home-margin-rank", "figure"),
        Output("hist-home-note", "children"),
        Output("hist-resid-note", "children"),
        Input("delta-elev", "value"),
    )
    def _update_global(delta_range: List[int]):
        df = load_df()

        # Filtre Δ altitude sur les matchs (utilisé pour les histos / scatter)
        if delta_range and "delta_elev" in df.columns:
            lo, hi = delta_range
            mask = df["delta_elev"].between(lo, hi, inclusive="both")
            df_filtered = df[mask]
        else:
            df_filtered = df

        # ----- Carte globale : 30 équipes, altitude + capacité -----
        arenas = _build_arenas_df(df)
        if not arenas.empty:
            elev_norm = (arenas["elev_m"] - arenas["elev_m"].min()) / (
                arenas["elev_m"].max() - arenas["elev_m"].min() + 1e-9
            )
            cap_norm = (arenas["capacity"] - arenas["capacity"].min()) / (
                arenas["capacity"].max() - arenas["capacity"].min() + 1e-9
            )
            arenas["indice_altitude_capacite"] = (elev_norm + cap_norm) / 2

            arenas_plot = arenas.copy()
            arenas_plot["team"] = arenas_plot["home_team"]
            arenas_plot["Altitude"] = arenas_plot["elev_m"]

            fig_map = px.scatter_geo(
                arenas_plot,
                lat="lat",
                lon="lon",
                color="indice_altitude_capacite",
                size="capacity",
                hover_name="arena",
                hover_data={
                    "team": True,
                    "capacity": True,
                    "Altitude": True,
                    "lat": False,
                    "lon": False,
                    "indice_altitude_capacite": False,
                },
                projection="natural earth",
                title="Carte des arènes NBA — altitude & capacité",
            )
            fig_map.update_coloraxes(colorbar_title="Indice altitude + capacité")
        else:
            fig_map = px.scatter_geo(title="Carte indisponible (colonnes manquantes).")

        fig_map.update_geos(
            showland=True,
            landcolor="#f8f9fa",
            showocean=True,
            oceancolor="#e8f1ff",
            bgcolor="white",
            scope="north america",
            showcountries=True,
            showcoastlines=True,
        )

        # ----- Histogramme marge brute domicile -----
        if "home_diff" in df_filtered.columns:
            df_home = df_filtered.dropna(subset=["home_diff"])
            fig_hist_home = px.histogram(
                df_home,
                x="home_diff",
                nbins=15,
                title="Histogramme — Marge à domicile (brute)",
            )
            fig_hist_home.update_layout(
                xaxis_title="Marge à domicile (pts)", yaxis_title="Fréquence"
            )
            if not df_home.empty:
                mean_home = df_home["home_diff"].mean()
                std_home = df_home["home_diff"].std(ddof=1)
                hist_home_note = f"{len(df_home)} matchs : moyenne {mean_home:.2f} pts, σ≈{std_home:.2f}."
                hist_home_note += (
                    " Avantage domicile positif."
                    if mean_home > 0
                    else " Avantage domicile limité."
                )
            else:
                hist_home_note = (
                    "Pas assez de données pour tracer la distribution brute."
                )
        else:
            fig_hist_home = px.histogram(
                title="Histogramme indisponible (home_diff manquant)."
            )
            hist_home_note = "Colonnes home_diff manquantes dans le dataset filtré."

        # ----- Histogramme marge résiduelle -----
        if "residual_margin" in df_filtered.columns:
            df_res = df_filtered.dropna(subset=["residual_margin"])
            fig_hist_resid = px.histogram(
                df_res,
                x="residual_margin",
                nbins=25,
                title="Histogramme — Marge résiduelle (ajustée Elo)",
            )
            fig_hist_resid.update_layout(
                xaxis_title="Marge résiduelle (pts)", yaxis_title="Fréquence"
            )
            if not df_res.empty:
                mean_res = df_res["residual_margin"].mean()
                std_res = df_res["residual_margin"].std(ddof=1)
                hist_resid_note = f"Résiduel moyen {mean_res:.2f} pts (σ≈{std_res:.2f}) sur {len(df_res)} matchs."
                hist_resid_note += (
                    " L'Elo annule presque tout l'avantage."
                    if abs(mean_res) < 0.5
                    else " La géographie laisse un avantage persistant."
                )
            else:
                hist_resid_note = "Pas assez de résidus pour cette sélection."
        else:
            fig_hist_resid = px.histogram(
                title="Histogramme indisponible (résiduel non calculé)."
            )
            hist_resid_note = (
                "Colonnes residual_margin manquantes dans le dataset filtré."
            )

        # ----- Scatter altitude vs marge résiduelle -----
        if {"elev_m", "residual_margin"}.issubset(df_filtered.columns):
            df_scat = df_filtered.dropna(subset=["elev_m", "residual_margin"])
            if not df_scat.empty:
                fig_scat = px.scatter(
                    df_scat,
                    x="elev_m",
                    y="residual_margin",
                    trendline="ols",
                    title="Résiduel (Elo) vs Altitude (m)",
                )
                fig_scat.update_layout(
                    xaxis_title="Altitude de l'arène (m)",
                    yaxis_title="Marge résiduelle (pts, ajustée Elo)",
                )
            else:
                fig_scat = px.scatter(
                    title="Scatter indisponible (données insuffisantes après filtre)."
                )
        else:
            fig_scat = px.scatter(title="Scatter indisponible (colonnes manquantes).")

        # ----- Classement marges domicile par équipe -----
        team_margins = pd.DataFrame()
        if {"home_team", "home_diff"}.issubset(df_filtered.columns):
            df_team = df_filtered.dropna(subset=["home_team", "home_diff"])
            if not df_team.empty:
                team_margins = df_team.groupby("home_team", as_index=False)[
                    "home_diff"
                ].mean()
                team_margins_sorted = team_margins.sort_values(
                    "home_diff", ascending=False
                )
                fig_team_bar = px.bar(
                    team_margins_sorted,
                    x="home_team",
                    y="home_diff",
                    title="Marge moyenne à domicile par franchise",
                )
                fig_team_bar.update_layout(
                    xaxis_title="Équipe (domicile)",
                    yaxis_title="Marge moyenne (pts)",
                )
            else:
                fig_team_bar = px.bar(
                    title="Graphique indisponible (aucune donnée domicile)."
                )
        else:
            fig_team_bar = px.bar(
                title="Graphique indisponible (colonnes home_team/home_diff manquantes)."
            )

        # ----- Mêler classement (victoires) et marge domicile -----
        standings = None
        if {
            "home_team",
            "away_team",
            "home_pts",
            "away_pts",
            "home_diff",
        }.issubset(df_filtered.columns):
            df_games = df_filtered.dropna(
                subset=["home_team", "away_team", "home_pts", "away_pts", "home_diff"]
            )
            if not df_games.empty:
                df_games = df_games.copy()
                df_games["home_win"] = (
                    df_games["home_pts"] > df_games["away_pts"]
                ).astype(int)
                df_games["away_win"] = (
                    df_games["away_pts"] > df_games["home_pts"]
                ).astype(int)

                home_wins = df_games.groupby("home_team")["home_win"].sum()
                away_wins = df_games.groupby("away_team")["away_win"].sum()
                total_wins = (
                    home_wins.add(away_wins, fill_value=0)
                    .rename_axis("team")
                    .reset_index(name="wins")
                )

                if not team_margins.empty:
                    margins_for_merge = team_margins.rename(
                        columns={"home_team": "team", "home_diff": "home_margin"}
                    )
                else:
                    margins_for_merge = pd.DataFrame(columns=["team", "home_margin"])

                standings = total_wins.merge(margins_for_merge, on="team", how="left")
                standings["home_margin"] = standings["home_margin"].astype(float)
                standings = standings.dropna(subset=["home_margin"])

        if standings is not None and not standings.empty:
            standings = standings.sort_values("wins", ascending=False)
            standings["rank"] = (
                standings["wins"].rank(method="dense", ascending=False).astype(int)
            )
            standings["text_pos"] = np.where(
                standings["rank"] % 2 == 0, "bottom center", "top center"
            )

            fig_margin_rank = px.scatter(
                standings,
                x="rank",
                y="home_margin",
                color="wins",
                size="wins",
                hover_name="team",
                text="team",
                title="Marge domicile vs nombre total de victoires",
                color_continuous_scale="deep",
            )
            if fig_margin_rank.data:
                fig_margin_rank.data[0].textposition = standings["text_pos"].tolist()
            fig_margin_rank.update_traces(
                textfont={"size": 10, "color": "#0f172a"},
                marker={"line": {"width": 0.5, "color": "#0f172a"}},
                cliponaxis=False,
            )
            fig_margin_rank.update_layout(
                xaxis_title="Classement (1 = plus de victoires)",
                yaxis_title="Marge moyenne à domicile (pts)",
                paper_bgcolor="#f8fafc",
                plot_bgcolor="#e2e8f0",
                coloraxis_colorbar=dict(title="Victoires", tickcolor="#0f172a"),
                xaxis=dict(showgrid=True, gridcolor="#cbd5f5"),
                yaxis=dict(showgrid=True, gridcolor="#cbd5f5"),
            )

            table_style = {
                "width": "100%",
                "borderCollapse": "collapse",
            }
            cell_style = {
                "padding": "6px 10px",
                "borderBottom": "1px solid #e5e7eb",
                "fontSize": "12px",
                "color": "#1f2937",
            }
            header_cell_style = {**cell_style, "fontWeight": 600, "fontSize": "13px"}

            header = html.Thead(
                html.Tr(
                    [
                        html.Th(
                            "Rang", style={**header_cell_style, "textAlign": "center"}
                        ),
                        html.Th(
                            "Équipe", style={**header_cell_style, "textAlign": "left"}
                        ),
                        html.Th(
                            "Victoires totales",
                            style={**header_cell_style, "textAlign": "center"},
                        ),
                    ]
                )
            )

            body_rows = []
            for idx, row in enumerate(standings.itertuples()):
                row_background = "#f9fafb" if idx % 2 else "#ffffff"
                body_rows.append(
                    html.Tr(
                        [
                            html.Td(
                                int(row.rank),
                                style={
                                    **cell_style,
                                    "textAlign": "center",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Td(
                                row.team, style={**cell_style, "textAlign": "left"}
                            ),
                            html.Td(
                                int(row.wins),
                                style={
                                    **cell_style,
                                    "textAlign": "center",
                                    "fontVariantNumeric": "tabular-nums",
                                },
                            ),
                        ],
                        style={"background": row_background},
                    )
                )

            wins_ranking_content = html.Div(
                html.Table([header, html.Tbody(body_rows)], style=table_style),
                style={
                    "background": "white",
                    "border": "1px solid #e5e7eb",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "marginBottom": "12px",
                },
            )
        else:
            fig_margin_rank = px.scatter(
                title="Graphique indisponible (classement non calculé)."
            )
            wins_ranking_content = html.Div(
                "Classement indisponible (données victoires manquantes).",
                style={"fontSize": "12px", "color": "#6b7280", "margin": "8px 0"},
            )

        return (
            fig_map,
            fig_hist_home,
            fig_hist_resid,
            fig_scat,
            fig_team_bar,
            wins_ranking_content,
            fig_margin_rank,
            hist_home_note,
            hist_resid_note,
        )

    # ----- Dropdowns duel -----
    @app.callback(
        Output("team-1", "options"),
        Output("team-2", "options"),
        Input("map", "figure"),  # juste pour déclencher après chargement
    )
    def _populate_duel_dropdowns(_fig):
        df = load_df()
        if "home_team" not in df.columns:
            return [], []
        teams = sorted(df["home_team"].dropna().unique())
        options = [{"label": t, "value": t} for t in teams]
        return options, options

    # ----- Duel entre deux équipes -----
    @app.callback(
        Output("duel-summary", "children"),
        Output("duel-bar", "figure"),
        Input("team-1", "value"),
        Input("team-2", "value"),
    )
    def _update_duel(team1: str | None, team2: str | None):
        if not team1 or not team2:
            return (
                html.Div(
                    "Sélectionnez une équipe à domicile puis un adversaire en déplacement."
                ),
                px.bar(title="Duel indisponible"),
            )
        if team1 == team2:
            return (
                html.Div(
                    "L'équipe domicile et l'équipe extérieure doivent être différentes."
                ),
                px.bar(title="Duel indisponible"),
            )

        df = load_df()
        required = {"home_team", "away_team", "home_diff"}
        if not required.issubset(df.columns):
            return (
                html.Div("Colonnes manquantes pour le duel."),
                px.bar(title="Duel indisponible"),
            )

        mask = (df["home_team"] == team1) & (df["away_team"] == team2)
        dfd = df[mask].copy()
        if dfd.empty:
            return (
                html.Div(
                    "Aucun match de saison régulière où "
                    f"{team1} recevaient {team2} n'est disponible en 2021-22."
                ),
                px.bar(title="Duel indisponible"),
            )

        dfd = dfd.sort_values("date")
        dfd["date_str"] = pd.to_datetime(dfd["date"]).dt.strftime("%Y-%m-%d")
        mean_home = dfd["home_diff"].mean()

        match_items = [
            html.Li(
                f"{row.date_str} : {team1} {int(row.home_pts)}-{int(row.away_pts)} {team2} • marge domicile = {row.home_diff:.2f} pts"
            )
            for row in dfd.itertuples()
        ]

        summary = html.Div(
            [
                html.P(
                    f"{len(dfd)} match(s) joués à {team1} contre {team2} en 2021-22",
                    style={"fontWeight": 600},
                ),
                html.Ul(match_items, style={"paddingLeft": "18px"}),
                html.P(
                    f"Marge moyenne à domicile : {mean_home:.2f} pts",
                    style={"marginTop": "6px"},
                ),
            ]
        )

        fig = px.bar(
            dfd,
            x="date_str",
            y="home_diff",
            title=f"Marge par match — {team1} (domicile) vs {team2}",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Marge domicile (pts)",
        )

        return summary, fig
