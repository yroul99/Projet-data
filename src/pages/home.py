# src/pages/home.py
from __future__ import annotations
from pathlib import Path
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
    {"label": "Marge à domicile (brute)",            "value": "home_diff"},
    {"label": "Marge résiduelle (ajustée Elo)",      "value": "residual_margin"},
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

        # --- Carte (agrégation par arène) ---
        if {"arena", "lat", "lon"}.issubset(df.columns):
            gm = (
                df.dropna(subset=["lat", "lon"])
                  .groupby(["arena", "lat", "lon", "capacity"], as_index=False)[metric]
                  .mean()
            )

            # Couleurs centrées si résiduel
            midpoint = 0.0 if metric == "residual_margin" else gm[metric].mean()

            fig_map = px.scatter_geo(
                gm,
                lat="lat",
                lon="lon",
                hover_name="arena",
                color=metric,
                size="capacity",
                size_max=40,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=midpoint,
                title=f"Carte — moyenne {metric} par arène",
            )
            fig_map.update_geos(showcountries=True, showcoastlines=True, fitbounds="locations")
            fig_map.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" +
                              f"{metric}: %{color:.2f}<br>" +
                              "Capacité: %{marker.size}<extra></extra>"
            )
        else:
            fig_map = px.scatter_geo(title="Carte indisponible (colonnes lat/lon manquantes)")

        # --- Histogramme ---
        series = df[metric].dropna()
        fig_hist = px.histogram(series, nbins=30, title=f"Histogramme — {metric}")
        fig_hist.update_xaxes(title=metric)
        fig_hist.update_yaxes(title="fréquence")

        return fig_map, fig_hist
