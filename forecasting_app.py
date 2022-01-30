import pandas as pd
import streamlit as st

from models import *
from plots import plot_forecast

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
    start_date = st.slider(
        "When to do the prediction?",
        value=input_df.index[-1].to_pydatetime(),
        format="MM/DD/YY -hh:mm",
        min_value=input_df.index[0].to_pydatetime(),
    )

    df = input_df.loc[input_df.index <= start_date]

    col = st.sidebar.selectbox("Select a column", df.columns)
    model_name = st.sidebar.selectbox("Select a model", ["ARIMA"])
    freq = st.sidebar.text_input("Enter frequency as string e.g. 1D", value="1D")
    horizon = st.sidebar.text_input("Enter horizon as string e.g. 1D", value="1D")
    p = st.sidebar.number_input("Auto Regressive(p)", value=1)
    d = st.sidebar.number_input("Difference (d)", value=0)

    if model_name == "ARIMA":

        ts = df[col].dropna().resample(freq).mean()
        model = ARIMAModel(horizon=horizon, freq=freq, order=(p, d, 0))
        model.fit(ts)
        prediction = model.predict()
    else:
        model_name = None

    fig = plot_forecast(df[col], prediction)
    st.plotly_chart(fig)

else:
    st.text("To use this tool, just upload a csv file with tabular format")
