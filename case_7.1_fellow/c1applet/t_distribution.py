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
from scipy.stats import nct

JupyterDash.infer_jupyter_proxy_config()

app = JupyterDash(__name__)

# Create server variable with Flask server object for use with gunicorn
server = app.server

app.layout = html.Div(
    [
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
                        html.P("Number of draws:"),
                        dcc.Slider(
                            id="number-draws-right-slider",
                            min=1,
                            max=200,
                            value=10,
                            step=5,
                            marks={
                                20: "20",
                                40: "40",
                                60: "60",
                                80: "80",
                                100: "100",
                                120: "120",
                                140: "140",
                                160: "160",
                                180: "180",
                                200: "200",
                            },
                        ),
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
        Input("number-draws-right-slider", "value"),
        Input("sample-size-left-slider", "value"),
    ],
)
def update_figure(number_draws, sample_size):
    df = pd.read_csv("data/natural_reserve_a.csv", usecols=["HT"])
    means_list = []
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
    fig.add_vline(
        x=np.mean(means_list),
        line_width=3,
        line_color="red",
        annotation_text="Mean of the sample means",
        annotation_position="top right",
        annotation_font_color="red",
    )

    fig.add_vline(
        x=df["HT"].mean(),
        line_width=3,
        line_color="#8a450c",
        annotation_text="Population mean μ",
        annotation_position="top left",
        annotation_font_color="#8a450c",
    )

    fig.update_layout(title_text="The t distribution approximation in blue")

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

    return fig
