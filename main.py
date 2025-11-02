from dash import Dash
from src.pages.home import layout, register_callbacks

app = Dash(__name__)
app.title = "NBA 2021-22 — Avantage du terrain"

# IMPORTANT : layout est un objet html.Div (pas une fonction)
app.layout = layout

register_callbacks(app)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)  # debug=True pour l’auto-reload
