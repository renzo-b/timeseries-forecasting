import pandas as pd


class BaseModel:
    """Base forecasting class"""

    def __init__(self):
        self.resample_slide = None

    def infer_freq(self, ts):
        """infers frequency of timeseries"""
        freq = pd.infer_freq(ts.index)

        if freq is None:
            raise ValueError(
                "Could not infer frequency from Datetime. Consider resampling the df"
            )

        if freq[0] not in "0123456789":  # add a digit for pd.to_timedelta to work
            freq = f"1{freq}"

        self.resample_slide = freq

        return self.resample_slide

