from sklearn.preprocessing import StandardScaler


class Scaler:

    def __init__(self, scale_cols):
        self.scale_cols = scale_cols
        self.scaler = StandardScaler()

    def fit(self, X):
        X = X.copy()
        self.scaler.fit(X[self.scale_cols])
        return self


    def transform(self, X):
        X = X.copy()
        X[self.scale_cols] = self.scaler.transform(X[self.scale_cols])
        return X


    def fit_transform(self, X):
        return self.fit(X).transform(X)