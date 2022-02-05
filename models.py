from pandas import to_timedelta
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler


class ModelARIMA:
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

    def fit(self, x):
        """
        x: array or pd.Series
            Input time series data
        """
        self.fitted_model = ARIMA(endog=x, order=self.order, freq=self.freq).fit()
        self.start = x.index[-1] + to_timedelta(self.freq)
        self.end = x.index[-1] + to_timedelta(self.horizon)

    def predict(self):
        """Performs and returns prediction"""
        prediction = self.fitted_model.predict(start=self.start, end=self.end)
        return prediction


class ClassLSTM(nn.Module):
    """Wrapper around pytorch lstm model for compatibility with models from other 
    libraries
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(ClassLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


class ModelLSTM:
    def __init__(self, seq_length, input_size, hidden_size, num_epochs, learning_rate):
        self.lstm = None
        self.train = None
        self.seq_length = seq_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def fit(self, ts):
        """ts: pd.Series"""
        scaler = StandardScaler()
        x = scaler.fit_transform(ts.values.reshape(-1, 1))

        train, test = sliding_windows(x, self.seq_length)

        # convert to tensors
        train = Variable(torch.Tensor(np.array(train)))
        test = Variable(torch.Tensor(np.array(test)))

        self.train = train

        self.lstm = ClassLSTM(self.input_size, self.hidden_size, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            outputs = self.lstm(train)
            optimizer.zero_grad()

            loss = criterion(outputs, test)

            loss.backward()

            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    def predict(self):
        return self.lstm(self.train)


def sliding_windows(data, seq_length):
    """
    Inputs
    ------
    data: array of shape (n_samples, 1)
        data to split
    
    seq_length: int
        length of sequence

    Returns
    -------
    train: array of shape (n_sequences, seq_length, 1)
        training data
    
    test: array of shape (n_sequences, 1)
        test data
    """
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)
