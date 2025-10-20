from __future__ import annotations
import pandas as pd
from dash import html, dcc, Output, Input, callback
import plotly.express as px
from config import CLEAN_FILE

def _load_df() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_FILE)
    # colonnes calculées si absentes
    if "home_diff" not in df.columns:
        df["home_diff"] = df["home_pts"] - df["away_pts"]
    if "points_total" not in df.columns:
        df["points_total"] = df["home_pts"] + df["away_pts"]
    return df

def layout():
    df = _load_df()
    metrics = ["home_diff", "points_total"]
    return html.Div([
        html.H2("NBA 2021-22 — Home Advantage (Altitude & Capacity)"),
        dcc.Dropdown(
            id="metric",
            options=[{"label":"Home point differential","value":"home_diff"},
                     {"label":"Total points","value":"points_total"}],
            value="home_diff", clearable=False
        ),
        dcc.Graph(id="hist"),
        dcc.Graph(id="map"),
    ], style={"maxWidth":"1100px","margin":"0 auto","padding":"16px"})

@callback(
    Output("hist","figure"),
    Output("map","figure"),
    Input("metric","value"),
)
def update(metric: str):
    df = _load_df()

    # Histogramme (variable numérique non catégorielle)
    fig_h = px.histogram(df, x=metric, nbins=30,
                         title=f"Distribution of {metric} — Season 2021-22")
    fig_h.update_layout(xaxis_title=metric, yaxis_title="Count")

    # Carte: agrégation par arène
    cols_needed = {"arena","lat","lon"}
    if cols_needed.issubset(df.columns):
        agg = (df
               .groupby(["arena","lat","lon","capacity"], as_index=False)["home_diff"]
               .mean())
        fig_m = px.scatter_geo(
            agg, lat="lat", lon="lon", hover_name="arena",
            size="capacity", color="home_diff", color_continuous_scale="RdBu",
            title="Arenas — Avg home diff (color) & Capacity (size)"
        )
        fig_m.update_geos(fitbounds="locations", showcountries=True, showcoastlines=True)
    else:
        # fallback si pas encore de géoloc
        fig_m = px.scatter_geo(title="Arenas map — add arenas to enable")
    return fig_h, fig_m
