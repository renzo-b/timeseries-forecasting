from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from models import *
from utils import *

model = ARIMAModel(horizon="3D", freq="1D", order=(5, 1, 0))

numdays = 100
data_range = np.arange(numdays)
datetimes = [datetime(2020, 1, 1) + timedelta(days=x) for x in range(numdays)]
ts = pd.Series(data_range, index=datetimes)


cv_results = cross_validate(ts, model=model, n_splits=3, test_size=3,)
