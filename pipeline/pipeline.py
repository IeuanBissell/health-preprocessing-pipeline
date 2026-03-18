class PreprocessingPipeline:

    def __init__(self, missing_value_handler, numerical_scaler, categorical_encoder, feature_engineer=None):
        self.missing_handler = missing_value_handler
        self.scaler = numerical_scaler
        self.encoder = categorical_encoder
        self.feature_engineer = feature_engineer

    def fit(self, X):
        x_missing = self.missing_handler.fit_transform(X)
        if self.feature_engineer is not None:
            x_missing = self.feature_engineer.transform(x_missing)
        x_scaled = self.scaler.fit_transform(x_missing)
        self.encoder.fit(x_scaled)
        return self

    def transform(self, X):
        X = self.missing_handler.transform(X)
        if self.feature_engineer is not None:
            X = self.feature_engineer.transform(X)
        X = self.scaler.transform(X)
        X = self.encoder.transform(X)
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)