import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

MISSING_PCT_THRESHOLD = 20
SKEW_THRESHOLD = 0.5
ROW_ZERO_THRESHOLD = 2


class RevisedMissingValueHandler:

    def __init__(self, drop_cols, impute_cols, zero_cols, knn_neighbors=5):
        self.drop_cols = drop_cols
        self.impute_cols = impute_cols
        self.zero_cols = zero_cols
        self.knn_neighbors = knn_neighbors
        self.impute_values = {}

    def fit(self, X):
        X = X.copy()
        X = X.drop(self.drop_cols, axis=1)
        X = X.loc[(X[self.zero_cols] == 0).sum(axis=1) < ROW_ZERO_THRESHOLD]

        for col in self.zero_cols:
            X[col] = X[col].replace(0, np.nan)

        for col in self.impute_cols:
            missing_pct = (X[col].isna().sum()/X.shape[0])*100
            if missing_pct > MISSING_PCT_THRESHOLD:
                self.impute_values[col] = KNNImputer(n_neighbors=self.knn_neighbors)
                self.impute_values[col].fit(X[[col]])
            else:
                skew = X[col].skew()
                if abs(skew) > SKEW_THRESHOLD:
                    self.impute_values[col] = SimpleImputer(missing_values=np.nan, strategy='median')
                    self.impute_values[col].fit(X[[col]])
                else:
                    self.impute_values[col] = SimpleImputer(missing_values=np.nan, strategy='mean')
                    self.impute_values[col].fit(X[[col]])

        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.drop_cols, axis=1)
        X = X.loc[(X[self.zero_cols] == 0).sum(axis=1) < ROW_ZERO_THRESHOLD]

        for col in self.zero_cols:
            X[col] = X[col].replace(0, np.nan)

        for col in self.impute_cols:
            X[col] = self.impute_values[col].transform(X[[col]]).flatten()
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)