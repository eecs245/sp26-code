import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly
import plotly.figure_factory as ff
import plotly.express as px
import plotly.io as pio

# Preferred styles
pio.templates["pds"] = go.layout.Template(
    layout=dict(
        margin=dict(l=30, r=30, t=30, b=30),
        autosize=True,
        width=600,
        height=400,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        title=dict(x=0.5, xanchor="center"),
    )
)
pio.templates.default = "simple_white+pds"

# display options for numpy and pandas
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.set_option("display.max_rows", 7)
pd.set_option("display.max_columns", 8)
pd.set_option("display.precision", 5)

# Use plotly as default plotting engine
pd.options.plotting.backend = "plotly"

FONT_FAMILY = "Palatino"

def create_base_scatter(X_train, y_train):
    fig = (
        X_train.assign(Outcome=y_train.astype(str).replace({'0': 'no diabetes', '1': 'yes diabetes'}))
                .plot(kind='scatter', x='Glucose', y='BMI', color='Outcome', 
                      color_discrete_map={'no diabetes': 'orange', 'yes diabetes': 'blue'},
                      title='Relationship between Glucose, BMI, and Diabetes', size_max=7, size=[1] * len(X_train))
                .update_layout(width=700, height=500)
    )
    return fig

def show_decision_boundary(model, X_train, y_train, title='', grid_n=40):
    from plotly.subplots import make_subplots

    # Create grid for decision boundary
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)

    # X_train col 1 = glucose, col 2 = BMI (0-indexed: 0=glucose, 1=BMI)
    glucose = X_train[:, 0]
    bmi = X_train[:, 1]

    if not grid_n:
        grid_n = 40

    # grid_n = 400
    # model_name = model.__class__.__name__.lower()
    # if ("knn" in model_name) or (hasattr(model, "k") and hasattr(model, "p") and hasattr(model, "X_train_")):
    #     grid_n = 120

    tol = 0
    x_min, x_max = glucose.min() - tol, glucose.max() + tol
    y_min, y_max = bmi.min() - tol, bmi.max() + tol
    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create figure
    fig = make_subplots()
    
    # Add decision boundary
    fig.add_trace(go.Contour(
        x=xs,
        y=ys,
        z=Z,
        colorscale=[(0, 'orange'), (1, 'blue')],
        opacity=0.5,
        showscale=False
    ))

    # Add scatter points
    mask_no = (y_train == 0)
    fig.add_trace(go.Scatter(
        x=glucose[mask_no],
        y=bmi[mask_no],
        mode='markers',
        marker=dict(color='orange', size=8),
        name='no diabetes'
    ))
    
    mask_yes = (y_train == 1)
    fig.add_trace(go.Scatter(
        x=glucose[mask_yes],
        y=bmi[mask_yes],
        mode='markers',
        marker=dict(color='blue', size=8),
        name='yes diabetes'
    ))

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title='Glucose',
        yaxis_title='BMI',
        showlegend=True,
        legend=dict(font=dict(size=12)),
        width=700,
        height=500
    )

    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    return fig