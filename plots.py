import matplotlib.pyplot as plt
import plotly.graph_objs as go
import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_forecast(
    historical, actual, prediction, title=None, xlabel=None, ylabel="value"
):

    data = []
    data.append(
        go.Scatter(
            name="Historical",
            x=historical.index.values,
            y=historical.values,
            mode="lines+markers",
        )
    )
    data.append(
        go.Scatter(
            name="Actual", x=actual.index.values, y=actual.values, mode="lines+markers"
        )
    )
    data.append(
        go.Scatter(
            name="Predicted",
            x=prediction.index.values,
            y=prediction.values,
            mode="lines+markers",
        )
    )
    fig = go.Figure(data=data)

    return fig


def auto_corr_plot(ts, lags=None, figsize=(12, 7)):
    """
    Plots time series, auto correlation, partial auto correlation, and performs AD Fuller test
    """
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_vals = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_vals = plt.subplot2grid(layout, (1, 0))
    pacf_vals = plt.subplot2grid(layout, (1, 1))

    ts.plot(ax=ts_vals)
    plot_acf(ts, lags=lags, ax=acf_vals)
    plot_pacf(ts, lags=lags, ax=pacf_vals)
    plt.tight_layout()

    p_value = sm.tsa.stattools.adfuller(ts)[1]
    print(f"Dickey Fuller Test: {p_value:0.5f}")
    return fig
