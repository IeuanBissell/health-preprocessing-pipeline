import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


class CategoricalEncoder:

    def __init__(self, nominal_cols, ordinal_cols):
        self.nominal_cols = nominal_cols
        self.ordinal_cols = ordinal_cols
        self.ordinal = {}
        self.nominal = {}

    def fit(self, X):
        X = X.copy()
        for col, order in self.ordinal_cols.items():
            self.ordinal[col] = OrdinalEncoder(categories=[order])
            self.ordinal[col].fit(X[[col]])

        for col in self.nominal_cols:
            self.nominal[col] = OneHotEncoder(sparse_output=False)
            self.nominal[col].fit(X[[col]])

        return self

    def transform(self, X):
        X = X.copy()

        for col, order in self.ordinal_cols.items():
            X[col] = self.ordinal[col].transform(X[[col]]).flatten()

        for col in self.nominal_cols:
            encoded = self.nominal[col].transform(X[[col]])
            encoded_df = pd.DataFrame(encoded,
                                      columns=self.nominal[col].get_feature_names_out(),
                                      index=X.index)
            X = X.drop(col, axis=1)
            X = pd.concat([X, encoded_df], axis=1)

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)