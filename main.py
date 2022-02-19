from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from models import *
from utils import *

# data read
numdays = 100
data_range = np.arange(numdays)
datetimes = [datetime(2020, 1, 1) + timedelta(days=x) for x in range(numdays)]
ts = pd.Series(data_range, index=datetimes)
# ts = pd.read_csv("datasets/dummy_dataset.csv", parse_dates=["Feature 1"])
# ts = ts["Feature 1"]

# splitter
n_splits = 3
validate_size = 3
gap = 0
splitter = TimeSeriesSplit(n_splits=n_splits, test_size=validate_size, gap=gap)

# model
# model = ModelARIMA(horizon="3D", freq="1D", order=(5, 1, 0))
model = ModelLSTM(3, 3, 1, 3, 2000, 0.01)

# model validation
cv_results = cross_validate(ts, model=model, splitter=splitter)

