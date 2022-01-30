from pandas import to_timedelta
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """Wrapper around statsmodels ARIMA for compatibility 
    with models from other libraries
    """

    def __init__(self, horizon, order, freq):
        self.fitted_model = None
        self.order = order
        self.freq = freq
        self.horizon = horizon
        self.start = None
        self.end = None

    def fit(self, x):
        self.fitted_model = ARIMA(endog=x, order=self.order, freq=self.freq).fit()
        self.start = x.index[-1] + to_timedelta(self.freq)
        self.end = x.index[-1] + to_timedelta(self.horizon)

    def predict(self, x=None):
        prediction = self.fitted_model.predict(start=self.start, end=self.end)
        return prediction
