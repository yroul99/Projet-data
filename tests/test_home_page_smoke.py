"""Tests fumée pour vérifier que layout et callbacks Dash se construisent."""
from __future__ import annotations
from dash import Dash
from src.pages import home


def test_layout_builds():
    """Le layout de la page home doit exister et être accessible."""
    assert hasattr(home, "layout")
    assert home.layout is not None


def test_callbacks_register():
    """L'enregistrement des callbacks ne doit lever aucune exception."""
    app = Dash(__name__)
    home.register_callbacks(app)
