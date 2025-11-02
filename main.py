<<<<<<< HEAD
from dash import Dash
=======
﻿from dash import Dash
>>>>>>> 003b078 (Update: clean_data (altitude/B2B/Elo) + pages/home)
from src.pages.home import layout, register_callbacks

app = Dash(__name__)
app.title = "NBA 2021-22 — Avantage du terrain"

app.layout = layout
register_callbacks(app)

if __name__ == "__main__":
<<<<<<< HEAD
    app.run(host="0.0.0.0", port=8050, debug=True)  # ← nouveau: app.run(...)
=======
    app.run(host="0.0.0.0", port=8050, debug=True)
>>>>>>> 003b078 (Update: clean_data (altitude/B2B/Elo) + pages/home)
