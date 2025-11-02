# src/pages/home.py
from __future__ import annotations
import pandas as pd
import plotly.express as px
from dash import html, dcc, Output, Input
from config import CLEAN_FILE

def load_df() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_FILE)
    if "home_diff" not in df.columns:
        df["home_diff"] = df["home_pts"] - df["away_pts"]
    return df

METRIC_OPTIONS = [
    {"label": "Marge à domicile (brute)",       "value": "home_diff"},
    {"label": "Marge résiduelle (ajustée Elo)", "value": "residual_margin"},
]

layout = html.Div(
    [
        html.H2("NBA 2021-22 — Avantage du terrain"),
        dcc.Dropdown(
            id="metric",
            options=METRIC_OPTIONS,
            value="residual_margin",
            clearable=False,
            style={"width": "420px"}
        ),
        dcc.Graph(id="map"),
        dcc.Graph(id="hist"),
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "16px"},
)

def register_callbacks(app):
    @app.callback(
        Output("map", "figure"),
        Output("hist", "figure"),
        Input("metric", "value"),
    )
    def _update(metric: str):
        df = load_df()

        # -------- Carte (agrégation par arène) --------
        if {"arena", "lat", "lon"}.issubset(df.columns):
            gm = (
                df.dropna(subset=["lat", "lon"])
                  .groupby(["arena", "lat", "lon", "capacity"], as_index=False)[metric]
                  .mean()
            )
            fig_map = px.scatter_geo(
                gm,
                lat="lat", lon="lon",
                hover_name="arena",
                color=metric,
                size="capacity", size_max=40,
                color_continuous_scale="RdBu",
                title=f"Carte — moyenne {metric} par arène",
                hover_data={"capacity": True, metric:":.2f"},
            )
            fig_map.update_geos(showcountries=True, showcoastlines=True, fitbounds="locations")

            # Palette symétrique autour de 0 pour le résiduel
            if metric == "residual_margin":
                v = gm[metric].abs().quantile(0.95)
                fig_map.update_coloraxes(cmid=0, cmin=-v, cmax=v)
        else:
            fig_map = px.scatter_geo(title="Carte indisponible (colonnes lat/lon manquantes)")

        # -------- Histogramme --------
        fig_hist = px.histogram(df, x=metric, nbins=30, title=f"Histogramme — {metric}")
        fig_hist.update_layout(xaxis_title=metric, yaxis_title="Fréquence")

        return fig_map, fig_hist
