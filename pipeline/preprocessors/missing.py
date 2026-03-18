import numpy as np
from sklearn.impute import SimpleImputer

class MissingValueHandler:

    def __init__(self, drop_col, mean_col, median_col, zero_cols):
        self.drop_col = drop_col
        self.mean_col = mean_col
        self.median_col = median_col
        self.zero_cols = zero_cols
        self.impute_values = {}

    def fit(self, X):
        X = X.copy()
        X = X.drop(self.drop_col, axis=1)
        X = X.loc[(X[self.zero_cols] == 0).sum(axis=1) < 2]

        for col in self.zero_cols:
            X[col] = X[col].replace(0, np.nan)

        for col in self.mean_col:
            self.impute_values[col] = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.impute_values[col].fit(X[[col]])

        for col in self.median_col:
            self.impute_values[col] = SimpleImputer(missing_values=np.nan, strategy='median')
            self.impute_values[col].fit(X[[col]])

        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.drop_col, axis=1)
        X = X.loc[(X[self.zero_cols] == 0).sum(axis=1) < 2]

        for col in self.zero_cols:
            X[col] = X[col].replace(0, np.nan)

        for col in self.mean_col:
            X[col] = self.impute_values[col].transform(X[[col]]).flatten()

        for col in self.median_col:
            X[col] = self.impute_values[col].transform(X[[col]]).flatten()

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)