from __future__ import annotations
from dash import Dash, html
from config import CLEAN_FILE, DATA_CLEAN
from src.pages.home import layout

# S'assure que le dossier cleaned existe
DATA_CLEAN.mkdir(parents=True, exist_ok=True)

app = Dash(__name__)
app.layout = html.Div([ layout() ])

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
