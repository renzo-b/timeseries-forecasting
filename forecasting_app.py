import pandas as pd
import streamlit as st

from models import *
from plots import *

input_data = None


def reading_dataset():
    global dataset
    try:
        dataset = pd.read_excel(input_data, parse_dates=[0])
        dataset.set_index(dataset.columns[0], inplace=True)
    except ValueError:
        dataset = pd.read_csv(input_data, parse_dates=[0], infer_datetime_format=True)
        dataset.set_index(dataset.columns[0], inplace=True)
    return dataset


st.title("Time Series Forecasting 🔮")
input_data = st.sidebar.file_uploader("", type=[".xlsx", ".csv"])

if input_data:
    input_df = reading_dataset()

    ## general inputs
    start_date = st.slider(
        "When to do the prediction?",
        value=input_df.index[-1].to_pydatetime(),
        format="YY/MM/DD hh:mm",
        min_value=input_df.index[0].to_pydatetime(),
        max_value=input_df.index[-1].to_pydatetime(),
    )
    col = st.sidebar.selectbox("Select a column", input_df.columns)
    model_name = st.sidebar.selectbox("Select a model", ["ARIMA"])
    freq = st.sidebar.text_input("Enter frequency as string e.g. 1D", value="1D")
    horizon = st.sidebar.number_input("Enter horizon as integer", value=1)
    p = st.sidebar.number_input("Auto Regressive(p)", value=1)
    d = st.sidebar.number_input("Difference (d)", value=0)
    lags = st.sidebar.number_input("Lags", value=1)

    horizon = str(horizon * int(freq[:-1])) + freq[-1]

    ts = input_df[col].dropna().resample(freq).mean()

    historical = ts.loc[ts.index <= start_date]
    actual = ts.loc[
        (ts.index > start_date) & (ts.index <= start_date + pd.to_timedelta(horizon))
    ]

    if model_name == "ARIMA":
        model = ARIMAModel(horizon=horizon, freq=freq, order=(p, d, 0))
        model.fit(historical)
        prediction = model.predict()
    else:
        model_name = None

    fig = plot_forecast(historical=historical, actual=actual, prediction=prediction)
    st.plotly_chart(fig)

    st.subheader("ACF, PACF Plots")
    fig = auto_corr_plot(ts=historical, lags=lags)
    st.pyplot(fig)

else:
    st.text("To use this tool, just upload a csv file with tabular format")
