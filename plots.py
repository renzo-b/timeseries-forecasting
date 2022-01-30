import plotly.graph_objs as go


def plot_forecast(ts, prediction, title=None, xlabel=None, ylabel="value"):

    data =[]
    data.append(go.Scatter(
            name='Actual',
            x=ts.index.values,
            y=ts.values,
            mode='markers'
        ))
    data.append(go.Scatter(
        name='Predicted',
        x=prediction.index.values,
        y=prediction.values,
        mode='markers'
    ))
    fig = go.Figure(data=data)
    
    return fig
