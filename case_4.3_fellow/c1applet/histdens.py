from jupyter_dash import JupyterDash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.figure_factory as ff
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



def fig_to_uri(in_fig, close_all=True, **save_args):
    # Function adapted from https://github.com/4QuantOSS/DashIntro/blob/master/notebooks/Tutorial.ipynb
    """Save a figure as a URI
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    in_fig.clf()
    plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


JupyterDash.infer_jupyter_proxy_config()

app = JupyterDash(__name__)

# Create server variable with Flask server object for use with gunicorn
server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.P("Histogram bin width:"),
            dcc.Slider(
                id='binwidth',
                min=0.1,
                max=6,
                value=1.5,
                step=0.1
            ),
        ], className="six columns"),
        html.Div([
            html.P("Density plot bandwidth:"),
            dcc.Slider(
                id='bandwidth',
                min=0.1,
                max=0.8,
                value=0.25,
                step=0.01
            ),
        ], className="six columns"),

    ], className="row"),
    html.Div([
        html.Img(
            id='plot',
            style={'height':'60%', 'width':'60%'})], style={'textAlign': 'center'})
])


@app.callback(
    Output('plot', 'src'),
    [Input('binwidth', 'value'),
    Input('bandwidth', 'value'),])
def update_figure(binwidth,bandwidth):
    # Creating the dataset
    np.random.seed(10)
    v = np.random.normal(loc=15, scale=7, size=500)
    v = pd.Series(v)
    fig, ax = plt.subplots(1,1)
    sns.histplot(data=v, binwidth=binwidth, ax=ax, stat='density', color="#1F1B42")
    sns.kdeplot(data=v, bw_adjust = bandwidth, ax=ax, color="red")
    plt.figure(figsize=(25,20))
    out_uri = fig_to_uri(fig)
    return out_uri
