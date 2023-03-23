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
                        html.P("Sample size (n):"),
                        dcc.Slider(
                            id="sample-size-left-slider",
                            min=2,
                            max=300,
                            value=10,
                            step=1,
                            marks={100: "100", 200: "200", 300: "300"},
                        ),
                    ],
                    className="six columns",
                ),
                html.Div(
                    [
                        # html.Br(),
                        html.P("Number of draws: Many"),
                    ],
                    className="six columns",
                ),
            ],
            className="row",
        ),
        # Second row
        html.Div(
            [
                html.Div(
                    [
                        # html.Br(),
                        html.P("Significance level α:"),
                        dcc.Slider(
                            id="alpha",
                            min=0.0001,
                            max=1.0,
                            value=0.15,
                            step=0.01,
                            marks={
                                0.0: "0",
                                0.1: "0.1",
                                0.2: "0.2",
                                0.3: "0.3",
                                0.4: "0.4",
                                0.5: "0.5",
                                0.6: "0.6",
                                0.7: "0.7",
                                0.8: "0.8",
                                0.9: "0.9",
                                1: "1",
                            },
                        ),
                    ],
                    className="twelve columns",
                ),
            ],
            className="row",
        ),
        # Third row
        html.Div(
            [
                html.Div(
                    [
                        html.Br(),
                        html.Div(
                            id="results-panel-B",
                            style={
                                "color": "grey",
                                "fontSize": 16,
                                "text-align": "center",
                            },
                        ),
                        html.Div(
                            id="results-panel-C",
                            style={
                                "color": "grey",
                                "fontSize": 16,
                                "text-align": "center",
                            },
                        ),
                        html.Br(),
                    ],
                    className="twelve columns",
                ),
            ],
            className="row",
        ),
        # Graph
        dcc.Graph(id="graph-with-slider"),
    ]
)


@app.callback(
    [
        Output("graph-with-slider", "figure"),
        Output("results-panel-B", "children"),
        Output("results-panel-C", "children"),
    ],
    [
        Input("sample-size-left-slider", "value"),
        Input("alpha", "value"),
    ],
)
def update_figure(sample_size, alpha):
    df = pd.read_csv("data/natural_reserve_a.csv", usecols=["HT"])

    # T distribution plot

    dfree, nc = sample_size - 1, 0
    mu = df["HT"].mean()
    S = df["HT"].std()
    n = sample_size
    se = S / math.sqrt(n)
    x = np.linspace(
        nct.ppf(0.000000001, dfree, nc, loc=mu, scale=se),
        nct.ppf(0.999999999, dfree, nc, loc=mu, scale=se),
        10000,
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

    # critical value
    critical_value = stats.nct.ppf(1 - alpha, dfree, nc=nc, loc=mu, scale=se)

    # Filled plot to show area under the curve
    x_filled = np.delete(x, np.where(x < critical_value))

    pdf_filled = nct.pdf(x_filled, dfree, nc, loc=mu, scale=se)
    # np.zeros(len(x_filled))
    fig.add_trace(
        go.Scatter(
            x=x_filled,
            y=pdf_filled,
            mode="lines",
            name="significance level",
            fill="tozeroy",
            fillcolor="rgba(0,255,255,0.8)",
            line=dict(width=0),
        )
    )

    fig.layout.template = "plotly_white"
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[20, 100])

    # Vertical lines
    B = 62
    C = 69

    fig.add_vline(
        x=critical_value,
        line_width=3,
        line_color="rgba(0,255,255,0)",
        annotation_text="The critical value for α="
        + str(alpha)
        + " is "
        + str(round(critical_value, 1))
        + " feet.",
        annotation_position="top right",
        annotation_font_color="rgb(10,107,104)",
    )

    fig.add_vline(
        x=mu,
        line_width=3,
        line_color="#8a450c",
        annotation_text="Population mean μ",
        annotation_position="top",
        annotation_font_color="#8a450c",
    )

    fig.add_vline(
        x=B,
        line_width=3,
        line_color="grey",
        annotation_text="B",
        annotation_position="top",
        annotation_font_color="grey",
    )

    fig.add_vline(
        x=C,
        line_width=3,
        line_color="#eb3471",
        annotation_text="C",
        annotation_position="top",
        annotation_font_color="#eb3471",
    )

    fig.update_layout(
        title_text="The t distribution approximation in blue — the shaded area in cyan is the significance level"
    )

    # Results panel

    if B < critical_value:
        h0_B_result = "FAILED TO REJECT"
    else:
        h0_B_result = "REJECTED"

    if C < critical_value:
        h0_C_result = "FAILED TO REJECT"
    else:
        h0_C_result = "REJECTED"

    text_results_B = """Null hypothesis for sample mean B: {}.""".format(h0_B_result)
    text_results_C = """Null hypothesis for sample mean C: {}.""".format(h0_C_result)

    return fig, text_results_B, text_results_C
