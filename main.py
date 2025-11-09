# main.py
import os
from dash import Dash, html
from src.pages.home import layout as home_layout, register_callbacks

app = Dash(
    __name__,
    title="NBA Home Advantage",
    suppress_callback_exceptions=False,
)
app.layout = html.Div([home_layout])

# Enregistre tous les callbacks définis dans la page
register_callbacks(app)

# Expose l'objet WSGI pour gunicorn/render/heroku
server = app.server

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    app.run(debug=True, host="0.0.0.0", port=port)
