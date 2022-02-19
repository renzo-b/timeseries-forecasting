import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .base_models import BaseModel


class ModelARIMA(BaseModel):
    """Wrapper around statsmodels ARIMA for compatibility with models from other 
    libraries
    """

    def __init__(self, horizon, order, freq):
        self.fitted_model = None
        self.order = order
        self.freq = freq
        self.horizon = horizon
        self.start = None
        self.end = None

    def fit(self, input_data, label_columns=None):
        """
        input_data: pd.DataFrame
            Input time series data
        """
        self.fitted_model = ARIMA(endog=input_data, order=self.order, freq=self.freq).fit()
        self.start = input_data.index[-1] + pd.to_timedelta(self.freq)
        self.end = input_data.index[-1] + pd.to_timedelta(self.horizon)

    def predict(self, input_data=None):
        """
        Performs and returns predictions
        input_data: input timeseries. Not used in this method. Included for consistency.
        """           
        prediction = self.fitted_model.predict(start=self.start, end=self.end)
        return prediction
