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
            # Taille = capacité (avec fallback médiane si NaN)
            gm["cap_plot"] = gm["capacity"].fillna(gm["capacity"].median())

            fig_map = px.scatter_geo(
                gm,
                lat="lat", lon="lon",
                hover_name="arena",
                color=metric,
                size="cap_plot",
                size_max=20,                        # cercles globalement petits
                color_continuous_scale="RdBu",
                hover_data={"capacity": True, metric:":.2f"},
            )

            # contour + taille mini + légère transparence
            fig_map.update_traces(
                marker=dict(line=dict(color="rgba(30,30,30,0.65)", width=1.2),
                            sizemin=6, opacity=0.95)
            )

            # palette symétrique autour de 0 pour le résiduel
            if metric == "residual_margin":
                v = gm[metric].abs().quantile(0.95)
                fig_map.update_coloraxes(cmid=0, cmin=-v, cmax=v)

            # <<< déplacé en dehors du else : toujours appliquer ces styles >>>
            fig_map.update_geos(
                showland=True,  landcolor="#f8f9fa",
                showocean=True, oceancolor="#e8f1ff",
                bgcolor="white", scope="north america",
                showcountries=True, showcoastlines=True
            )
            fig_map.update_layout(coloraxis_colorscale=[
                [0.0,  "#b2182b"],
                [0.45, "#f4a582"],
                [0.5,  "#f7f7f7"],  # 0 bien blanc -> contraste avec la terre
                [0.55, "#92c5de"],
                [1.0,  "#2166ac"]
            ])



        # -------- Histogramme --------
        series = df[metric].dropna()
        fig_hist = px.histogram(series, nbins=30, title=f"Histogramme — {metric}")
        if metric == "residual_margin":
            vhist = series.abs().quantile(0.98)
            fig_hist.update_xaxes(range=[-vhist, vhist])
        fig_hist.update_layout(xaxis_title=metric, yaxis_title="Fréquence")

        return fig_map, fig_hist
