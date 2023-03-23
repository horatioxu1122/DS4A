import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash

JupyterDash.infer_jupyter_proxy_config()

app = JupyterDash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Br(),
                html.P("Select the variable:"),
                dcc.Dropdown(
                    id="x-var-dropdown",
                    options=[
                        {"label": "Education", "value": "education"},
                        {"label": "Seniority", "value": "seniority_years"},
                        {"label": "Performance", "value": "performance_score"},
                        {"label": "Job title", "value": "job_title"},
                    ],
                    value="education",
                ),
                dcc.Graph(id="filtered-boxplot"),
            ]
        )
    ]
)


@app.callback(
    Output("filtered-boxplot", "figure"),
    [
        Input("x-var-dropdown", "value"),
    ],
)
def update_figure(x_var):
    df = pd.read_csv("data/company_dataset.csv")

    if x_var == "education":
        color_boxes = px.colors.qualitative.Prism[0]
    elif x_var == "seniority_years":
        color_boxes = px.colors.qualitative.Prism[1]
    elif x_var == "performance_score":
        color_boxes = px.colors.qualitative.Prism[2]
    elif x_var == "job_title":
        color_boxes = px.colors.qualitative.Prism[3]

    fig = px.box(
        df, x=x_var, y="pay_yearly", color_discrete_sequence=[color_boxes], points="all"
    )
    fig.layout.template = "plotly_white"
    fig.update_layout(showlegend=False)
    return fig
