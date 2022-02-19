import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from windows import WindowManager

from .base_models import BaseModel


class ClassLSTM(nn.Module):
    """Wrapper around pytorch lstm model for compatibility with models from other 
    libraries
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(ClassLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return torch.transpose(out[:, None], 1, 2)


class ModelLSTM(BaseModel):
    def __init__(
        self,
        input_length,
        output_length,
        input_size,
        hidden_size,
        num_epochs,
        learning_rate,
    ):
        self.lstm = None
        self.scaler = None
        self.input_length = input_length
        self.output_length = output_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.window_splitter = None
        self.label_columns = None

    def fit(self, input_data, label_columns):
        """
        input_data: pd.DataFrame
            Input time series data
        """
        self.window_splitter = WindowManager(self.input_length, self.output_length, 0)
        self.label_columns = label_columns

        X, Y = self.window_splitter.get_training_windows(
            input_data, label_columns=label_columns
        )

        self.lstm = ClassLSTM(self.input_size, self.output_length, self.hidden_size, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            outputs = self.lstm(X)
            optimizer.zero_grad()

            loss = criterion(outputs, Y)

            loss.backward()

            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    def predict(self, input_data):
        """
        Performs and returns predictions
        ts: array or pd.Series
            Input time series data
        """
        resample_slide = self.infer_freq(input_data)

        X = self.window_splitter.get_prediction_window(input_data, self.label_columns)

        y = self.lstm(X)  # predict
        prediction = self.scaler.inverse_transform(
            y.detach().numpy().reshape(-1, 1)
        ).reshape(
            -1
        )  # unscale

        idx = [
            input_data.index[-1] + N * pd.to_timedelta(resample_slide)
            for N in range(1, len(prediction) + 1)
        ]

        return pd.Series(data=prediction, index=idx)
