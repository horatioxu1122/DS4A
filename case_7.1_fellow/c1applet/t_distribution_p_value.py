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
                        html.P("Move the x_i value around:"),
                        dcc.Slider(
                            id="x-slider",
                            min=13,
                            max=107,
                            value=30,
                            step=0.5,
                            marks={
                                20: "20 ft",
                                30: "30 ft",
                                40: "40 ft",
                                50: "50 ft",
                                60: "60 ft",
                                70: "70 ft",
                                80: "80 ft",
                                90: "90 ft",
                                900: "900 ft",
                                100: "100 ft",
                            },
                        ),
                    ],
                    className="twelve columns",
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
        Input("x-slider", "value"),
    ],
)
def update_figure(sample_size, x_slider):
    df = pd.read_csv("data/natural_reserve_a.csv", usecols=["HT"])
    means_list = []
    number_draws = 150
    for draw in range(0, number_draws):
        means_list.append(df["HT"].sample(sample_size).mean())

    means_list = [means_list]
    fig = create_distplot(
        means_list,
        group_labels=["Sampling distribution of the mean"],
        show_hist=True,
        show_rug=False,
        histnorm="probability",
        curve_type="kde",
        colors=["green"],
    )

    fig.layout.template = "plotly_white"
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[20, 100])

    fig.update_layout(
        title_text="The t distribution approximation in blue — the shaded area in yellow is the p-value"
    )

    # T distribution plot

    dfree, nc = sample_size - 1, 0
    mu = df["HT"].mean()
    S = df["HT"].std()
    n = sample_size
    se = S / math.sqrt(n)
    x = np.linspace(
        nct.ppf(0.0000001, dfree, nc, loc=mu, scale=se),
        nct.ppf(0.9999999, dfree, nc, loc=mu, scale=se),
        1000,
    )
    pdf = nct.pdf(x, dfree, nc, loc=mu, scale=se)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="t-distribution",
            line=dict(color="royalblue", width=4),
        )
    )

    # Filled plot to show area under the curve
    x_filled = np.delete(x, np.where(x < x_slider))

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
    p_value = 1 - stats.nct.cdf(x_slider, dfree, nc=nc, loc=mu, scale=se)

    fig.add_vline(
        x=x_slider,
        line_width=3,
        line_color="yellow",
        annotation_text="The p-value for "
        + str(x_slider)
        + " feet is "
        + str(round(p_value, 3)),
        annotation_position="top left",
        annotation_font_color="orange",
    )

    return fig
