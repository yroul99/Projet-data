# main.py
from dash import Dash, html
from src.pages.home import layout, register_callbacks

app = Dash(__name__)
# layout est déjà un composant (pas une fonction) -> on ne l'appelle pas
app.layout = html.Div([layout])

register_callbacks(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
