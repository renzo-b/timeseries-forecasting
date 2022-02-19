import pandas as pd
import streamlit as st

from models.arima import ModelARIMA
from models.pytorch_models import ModelLSTM
from plots import *

input_data = None

@st.cache
def reading_dataset():
    global dataset
    try:
        dataset = pd.read_excel(input_data, parse_dates=[0])
        dataset.set_index(dataset.columns[0], inplace=True)
    except ValueError:
        dataset = pd.read_csv(input_data, parse_dates=[0], infer_datetime_format=True)
        dataset.set_index(dataset.columns[0], inplace=True)
    return dataset

@st.cache
def acf_pacf_plots(ts, lags):
    return auto_corr_plot(ts=ts, lags=lags)


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
    model_name = st.sidebar.selectbox("Select a model", ["ARIMA", "LSTM"])
    freq = st.sidebar.text_input("Enter frequency as string e.g. 1D", value="1D")
    horizon_int = st.sidebar.number_input("Enter horizon as integer", value=1)
    horizon = str(horizon_int * int(freq[:-1])) + freq[-1]

    ts = input_df[col].resample(freq).mean().interpolate()
    historical = ts.loc[ts.index <= start_date]
    actual = ts.loc[
        (ts.index > start_date) & (ts.index <= start_date + pd.to_timedelta(horizon))
    ]

    # model fit and predict
    if model_name == "ARIMA":
        # ARIMA inputs
        p = st.sidebar.number_input("Auto Regressive(p)", value=1)
        d = st.sidebar.number_input("Difference (d)", value=0)
        
        # ARIMA fit and predict
        model = ModelARIMA(horizon=horizon, freq=freq, order=(p, d, 0))
        model.fit(historical)
        prediction = model.predict(historical)
    
    elif model_name == "LSTM":
        # LSTM inputs
        input_length = st.sidebar.number_input("Input length", value=3)
        hidden_size = st.sidebar.number_input("Hidden size", value=3)
        num_epochs = st.sidebar.number_input("Number epochs", value=500)
        learning_rate = st.sidebar.number_input("Learning rate", value=0.01)

        # LSTM fit and predict()
        model = ModelLSTM(
            input_length=input_length, 
            output_length=horizon_int, 
            input_size=1, 
            hidden_size=hidden_size, 
            num_epochs=num_epochs, 
            learning_rate=learning_rate,
        )
        model.fit(historical)
        prediction = model.predict(historical)
    else:
        model_name = None

    # plotting
    fig = plot_forecast(historical=historical, actual=actual, prediction=prediction)
    st.plotly_chart(fig)

    # st.subheader("ACF, PACF Plots")
    # lags = st.sidebar.number_input("Lags", value=1)
    # fig = acf_pacf_plots(ts=historical, lags=lags)
    # st.pyplot(fig)

else:
    st.text("To use this tool, just upload a csv file with tabular format")
