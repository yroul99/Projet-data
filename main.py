# main.py
from dash import Dash
from src.pages.home import layout, register_callbacks

app = Dash(__name__)
app.title = "NBA 2021-22 — Avantage du terrain"

# NE PAS appeler layout() : c'est un objet, pas une fonction
app.layout = layout

# callbacks
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)
