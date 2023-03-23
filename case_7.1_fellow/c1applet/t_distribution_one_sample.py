import math

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from plotly.figure_factory import create_distplot
from scipy import stats
from scipy.stats import nct

JupyterDash.infer_jupyter_proxy_config()

app = JupyterDash(__name__)

# Create server variable with Flask server object for use with gunicorn
server = app.server

app.layout = html.Div(
    [
        # First row
        html.Div(
            [
                html.Div(
                    [
                        # html.Br(),
                        html.P("Take ONE sample of this size (n):"),
                        dcc.Slider(
                            id="sample-size-left-slider",
                            min=3,
                            max=300,
                            value=50,
                            step=5,
                            marks={100: "100", 200: "200", 300: "300"},
                        ),
                    ],
                    className="six columns",
                ),
                html.Div(
                    [
                        # html.Br(),
                        html.P("Number of draws: 1"),
                    ],
                    className="six columns",
                ),
            ],
            className="row",
        ),
        dcc.Graph(id="graph-with-slider"),
    ]
)


@app.callback(
    Output("graph-with-slider", "figure"),
    [
        Input("sample-size-left-slider", "value"),
    ],
)
def update_figure(sample_size):
    df = pd.read_csv("data/natural_reserve_a.csv", usecols=["HT"])
    df_sample = df.sample(sample_size)

    # T distribution plot

    dfree, nc = sample_size - 1, 0
    mu = df["HT"].mean()
    S = df_sample["HT"].std()
    n = sample_size
    se = S / math.sqrt(n)
    x = np.linspace(
        nct.ppf(0.0000001, dfree, nc, loc=mu, scale=se),
        nct.ppf(0.9999999, dfree, nc, loc=mu, scale=se),
        1000,
    )
    pdf = nct.pdf(x, dfree, nc, loc=mu, scale=se)

    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="t-distribution",
            line=dict(color="royalblue", width=4),
        )
    )

    # Filled plot to show area under the curve
    xbar = df_sample["HT"].mean()
    x_filled = np.delete(x, np.where(x < xbar))
    pdf_filled = nct.pdf(x_filled, dfree, nc, loc=mu, scale=se)
    # np.zeros(len(x_filled))
    fig.add_trace(
        go.Scatter(
            x=x_filled,
            y=pdf_filled,
            mode="lines",
            name="t-distribution",
            fill="tozeroy",
            fillcolor="rgba(255,255,0,0.8)",
            line=dict(width=0),
        )
    )

    # p-value
    p_value = 1 - stats.nct.cdf(xbar, dfree, nc=nc, loc=mu, scale=se)

    fig.add_vline(
        x=xbar,
        line_width=3,
        line_color="yellow",
        annotation_text="The p-value for this sample mean ("
        + str(round(xbar, 1))
        + " feet) is "
        + str(round(p_value, 3)),
        annotation_position="top left",
        annotation_font_color="#d9ad0f",
    )

    fig.layout.template = "plotly_white"
    fig.update_layout(showlegend=False)
    fig.add_vline(
        x=mu,
        line_width=3,
        line_color="#8a450c",
        annotation_text="Population mean μ",
        annotation_position="bottom right",
        annotation_font_color="#8a450c",
    )

    fig.update_xaxes(range=[20, 100])
    fig.update_layout(
        title_text="The t distribution approximation in blue around the population mean"
    )

    return fig
