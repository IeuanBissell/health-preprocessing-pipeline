import pandas as pd

BMI_BINS = [0, 18.5, 24.9, 29.9, 100]
BMI_LABELS = ['Underweight', 'Healthy', 'Overweight', 'Obese']

class FeatureEngineer:

    def transform(self, X):
        X = X.copy()
        X['BMICategory'] = pd.cut(X['BMI'], bins=BMI_BINS, labels=BMI_LABELS)
        return X