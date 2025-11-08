# src/pages/home.py
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output

from config import CLEAN_FILE


# ---------------------------------------------------------------------
# Constantes / helpers
# ---------------------------------------------------------------------
SUMMARY_PATH = Path("data/cleaned/summary.json")

METRIC_OPTIONS = [
    {"label": "Marge à domicile (brute)",       "value": "home_diff"},
    {"label": "Marge résiduelle (ajustée Elo)", "value": "residual_margin"},
]
LABELS = {
    "home_diff": "Marge à domicile",
    "residual_margin": "Marge résiduelle (ajustée Elo)",
}

# Styles inline (remplace l'ancien html.Style)
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


def load_df() -> pd.DataFrame:
    """Charge le dataset nettoyé et ajoute home_diff si absent."""
    df = pd.read_csv(CLEAN_FILE)
    if "home_diff" not in df.columns:
        df["home_diff"] = df["home_pts"] - df["away_pts"]
    return df


def read_summary() -> dict:
    if SUMMARY_PATH.exists():
        try:
            return json.loads(SUMMARY_PATH.read_text())
        except Exception:
            pass
    return {}


def _fmt_ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci and len(ci) == 2 else "—"


def summary_block() -> html.Div:
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
            html.H4("Résumé (2021–22)", style={"marginBottom": "8px"}),
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
                        f"IC95% {_fmt_ci(elo.get('b0_ci95'))} • p={elo.get('b0_p', 0):.2e}",
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


def _slider_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """Bornes du RangeSlider basées sur delta_elev (arrondi au 100 m)."""
    if "delta_elev" in df.columns and df["delta_elev"].notna().any():
        vmin = int(math.floor(df["delta_elev"].min() / 100) * 100)
        vmax = int(math.ceil(df["delta_elev"].max() / 100) * 100)
        if vmin == vmax:
            vmin, vmax = vmin - 1000, vmax + 1000
        return vmin, vmax
    return -1700, 1700


# Calcul des bornes du slider à l'import du module
_DF0 = load_df()
EMIN, EMAX = _slider_bounds(_DF0)


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
layout = html.Div(
    [
        html.H2("NBA 2021-22 — Avantage du terrain"),
        summary_block(),

        dcc.Dropdown(
            id="metric",
            options=METRIC_OPTIONS,
            value="residual_margin",
            clearable=False,
            style={"width": "420px"},
        ),

        dcc.RangeSlider(
            id="delta-elev",
            min=EMIN,
            max=EMAX,
            step=50,
            value=[-300, 300],
            marks={v: f"{v} m" for v in range(EMIN, EMAX + 1, 300)},
            tooltip={"placement": "bottom", "always_visible": False},
        ),

        dcc.Loading(dcc.Graph(id="map"), type="dot"),
        dcc.Loading(dcc.Graph(id="hist"), type="dot"),
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "16px"},
)


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------
def register_callbacks(app):
    @app.callback(
        Output("map", "figure"),
        Output("hist", "figure"),
        Input("metric", "value"),
        Input("delta-elev", "value"),
    )
    def _update(metric: str, delta_range: list[int] | tuple[int, int]):
        df = load_df()

        # Filtre Δ altitude si dispo
        if delta_range and "delta_elev" in df.columns:
            lo, hi = delta_range
            df = df[(df["delta_elev"] >= lo) & (df["delta_elev"] <= hi)]

        # ---------- Carte (agrégation par arène) ----------
        have_cols = {"arena", "lat", "lon", metric}.issubset(df.columns)
        df_ll = df.dropna(subset=["lat", "lon"]) if have_cols else pd.DataFrame()

        if have_cols and not df_ll.empty:
            gm = (
                df_ll.groupby(["arena", "lat", "lon", "capacity"], as_index=False)[metric]
                .mean()
            )
            cap_med = float(gm["capacity"].median(skipna=True)) if "capacity" in gm else 18000.0
            gm["cap_plot"] = gm.get("capacity", pd.Series(index=gm.index)).fillna(cap_med)

            fig_map = px.scatter_geo(
                gm,
                lat="lat",
                lon="lon",
                hover_name="arena",
                color=metric,
                size="cap_plot",
                size_max=20,
                color_continuous_scale="RdBu",
                hover_data={"capacity": True, metric: ":.2f"},
                title=f"Carte — {LABELS.get(metric, metric)}",
            )

            # contour + taille mini + légère transparence
            fig_map.update_traces(
                marker=dict(
                    line=dict(color="rgba(30,30,30,0.65)", width=1.2),
                    sizemin=6,
                    opacity=0.95,
                )
            )

            # palette symétrique autour de 0 pour le résiduel
            if metric == "residual_margin":
                v = gm[metric].abs().quantile(0.95)
                fig_map.update_coloraxes(cmid=0, cmin=-v, cmax=v)
        else:
            fig_map = px.scatter_geo(title="Carte indisponible (colonnes manquantes ou filtre trop restrictif).")

        # Styles de carte (toujours appliqués)
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
        fig_map.update_layout(
            coloraxis_colorscale=[
                [0.0, "#b2182b"],
                [0.45, "#f4a582"],
                [0.5, "#f7f7f7"],  # 0 bien blanc
                [0.55, "#92c5de"],
                [1.0, "#2166ac"],
            ],
            coloraxis_colorbar=dict(title=LABELS.get(metric, metric)),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        # ---------- Histogramme ----------
        series = df[metric].dropna() if metric in df.columns else pd.Series([], dtype=float)
        fig_hist = px.histogram(
            series,
            nbins=30,
            title=f"Histogramme — {LABELS.get(metric, metric)}",
        )
        if metric == "residual_margin" and not series.empty:
            vhist = series.abs().quantile(0.98)
            fig_hist.update_xaxes(range=[-vhist, vhist])

        fig_hist.update_layout(
            xaxis_title=LABELS.get(metric, metric),
            yaxis_title="Fréquence",
            bargap=0.05,
        )

        return fig_map, fig_hist
