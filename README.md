# Health Data Preprocessing Pipeline

## Overview
A modular, automated preprocessing pipeline for tabular health datasets.
Designed to handle common data quality challenges found in clinical data
including missing values, inconsistent scaling, and categorical variables,
and produce clean, machine-learning-ready outputs in a reproducible and
reusable way.

Built as a portfolio project demonstrating skills relevant to health data
science research, using the Pima Indians Diabetes dataset as a case study.

## Installation
Clone the repository and install dependencies:

    git clone https://github.com/IeuanBissell/health-preprocessing-pipeline
    cd health-preprocessing-pipeline
    pip install -r requirements.txt

## Usage
Open notebooks/demo.ipynb for a full end-to-end walkthrough.

To use the pipeline on your own dataset:

    import pandas as pd
    from pipeline.preprocessors.revised_missing import RevisedMissingValueHandler
    from pipeline.preprocessors.numerical import Scaler
    from pipeline.preprocessors.categorical import CategoricalEncoder
    from pipeline.preprocessors.features import FeatureEngineer
    from pipeline.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(
        missing_value_handler=RevisedMissingValueHandler(
            drop_cols=['Insulin'],
            impute_cols=['Glucose', 'BloodPressure', 'SkinThickness', 'BMI'],
            zero_cols=['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
        ),
        numerical_scaler=Scaler(
            scale_cols=['Glucose', 'BloodPressure', 'SkinThickness',
                        'BMI', 'Pregnancies', 'DiabetesPedigreeFunction', 'Age']
        ),
        categorical_encoder=CategoricalEncoder(
            nominal_cols=[],
            ordinal_cols={
                'BMICategory': ['Underweight', 'Healthy', 'Overweight', 'Obese']
            }
        ),
        feature_engineer=FeatureEngineer()
    )

    df_clean = pipeline.fit_transform(df)

## Modules

### revised_missing.py — Missing Value Handler
Handles missing values encoded as zeros in clinical datasets.
Drops rows with two or more biologically impossible zeros.
Drops columns with excessive missingness (Insulin at 48%).
Automatically selects imputation strategy per column:
KNN imputation for columns with more than 20% missing values,
median imputation for skewed distributions,
and mean imputation for normal distributions.

### numerical.py — Scaler
Standardises numerical features using StandardScaler so no column
dominates model training due to scale differences.

### categorical.py — Categorical Encoder
Encodes categorical variables using ordinal encoding for ordered
categories such as BMI category, and one-hot encoding for nominal
categories such as blood type.

### features.py — Feature Engineer
Creates derived features before encoding. BMICategory is derived
from BMI using official NHS thresholds.

### pipeline.py — Preprocessing Pipeline
Orchestrates all modules in sequence:
1. Missing value handling
2. Feature engineering
3. Numerical scaling
4. Categorical encoding

## Design Decisions

**Automated strategy selection** means imputation strategy is chosen
automatically based on missingness percentage and distribution skewness,
reducing manual configuration and making the pipeline reusable across
different datasets.

**Fit and transform separation** follows the scikit-learn transformer
pattern where fit() learns statistics from training data only and
transform() applies them to any dataset, preventing data leakage.

**Modular architecture** means each preprocessing step is an independent,
testable module. Steps can be used individually or combined in the pipeline.

**NHS BMI thresholds** means BMI categorisation uses official WHO ranges
rather than arbitrary bins, ensuring clinical relevance.
Source: https://www.diabetes.co.uk/bmi.html
## Limitations
SkinThickness has 29.5% missing values. KNN imputation with high k
converges toward the median, limiting improvement over simple imputation.
BMI distribution shows some skew after imputation due to high missingness.
The pipeline is configured for the Pima dataset, and column names and
strategies would need updating for different datasets.

## Dataset
Pima Indians Diabetes Database
Source: Kaggle at https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
Original source: UCI Machine Learning Repository
768 patients, 9 features, binary diabetes outcome.
