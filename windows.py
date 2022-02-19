import numpy as np
import torch


class WindowManager:
    def __init__(self, input_width, label_width, shift, to_tensor=False):

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        self.to_tensor = to_tensor

    def _slide_windows(self, data, label_columns):
        """
        Inputs
        ------
        data: array 
            data to split
        
        input_length: int
            length of input sequence

        output_length; int
            length of output sequence

        Returns
        -------
        input_values: array 
            training data
        
        label_values: array 
            test data
        """
        data = data[label_columns]

        input_values = []
        label_values = []

        for i in range(
            len(data) - self.input_width - self.label_width - self.shift + 1
        ):
            _input_values = data[i : (i + self.input_width)]
            _label_values = data[
                (i + self.input_width + self.shift) : (
                    i + self.input_width + self.label_width + self.shift
                )
            ]
            input_values.append(_input_values)
            label_values.append(_label_values)

        return np.array(input_values), np.array(label_values)

    def get_training_windows(self, data, label_columns):
        input_values, label_values = self._slide_windows(data, label_columns)

        if self.to_tensor:
            input_values = torch.tensor(input_values, dtype=torch.float)
            label_values = torch.tensor(label_values, dtype=torch.float)

        print("Slice windows shape")
        print("Input: ", input_values.shape)
        print("Labels: ", label_values.shape)

        return input_values, label_values

    def get_prediction_window(self, data, label_columns):
        data = data[label_columns]
        input_values = []

        start = len(data) - self.input_width
        _input_values = data[start : (start + self.input_width)]
        input_values.append(_input_values)

        prediction_window = np.array(input_values)
        
        if self.to_tensor:
            prediction_window = torch.tensor(prediction_window, dtype=torch.float)

        print("Prediction window: ", prediction_window)

        return prediction_window

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_width}",
                f"Label indices: {self.label_width}",
                f"Shift: {self.shift}",
            ]
        )
