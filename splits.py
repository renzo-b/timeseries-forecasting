import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class SplitManager:
    """This class is meant to be used as part of cross validation to pass the dates for 
    train, validation, and test."""

    def __init__(self, df, n_validation_splits, validations_per_split, test_pct=None):
        self.df = df
        self.n_validation_splits = n_validation_splits
        self.validations_per_split = validations_per_split
        self.test_pct = test_pct
        self.has_test = None

    def __split_indexes(self):
        """Splits the data into train, validation, and test splits.
        
        Returns
        -------
        train_validate_idx: array
            indices including train and validate

        test_idx: array
            indices including test
        """
        n_rows = len(self.df)

        if self.test_pct:
            if self.test_pct > 1:
                raise ValueError("test_pct must be a value less or equal to 1")

            train_validate_pct = 1 - self.test_pct
            train_validate_idx = self.df.index[: int(train_validate_pct * n_rows)]
            test_idx = self.df.index[int(self.test_pct * n_rows) :]
            self.has_test = True

        else:
            train_validate_idx = self.df.index
            test_idx = np.empty(0)
            self.has_test = False

        return train_validate_idx, test_idx

    def get_train_validate(self):
        """
        Returns
        -------
        Returns a generator object

        Example:
        for train, validate in SplitManager.get_train_validate():
            ...
        """
        train_validate_idx, __ = self.__split_indexes()

        splitter = TimeSeriesSplit(
            n_splits=self.n_validation_splits,
            test_size=self.validations_per_split,
            gap=0,
        )

        return splitter.split(train_validate_idx)

    def get_test(self):
        """
        Returns
        -------
        test_idx: array
            test indices        
        """
        __, test_idx = self.__split_indexes()
        return test_idx

