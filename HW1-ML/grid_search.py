import polars as pl
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

import sklearn
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from sklearn.experimental import enable_iterative_imputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import IterativeImputer, SimpleImputer

import scipy.stats as scpt

import numpy as np


train_data_base = pl.read_csv("data/train.csv")
test_data_base = pl.read_csv("data/test.csv")


def transform_datetime_to_int(df: pl.DataFrame):
    return df.with_columns(
        pl.col("PolicyStart").str.to_datetime().dt.date().cast(pl.Int64),
    ).with_columns(
        pl.col(pl.String).str.to_lowercase(),
    )


train_data = transform_datetime_to_int(train_data_base)
test_data = transform_datetime_to_int(test_data_base).with_columns(
    pl.lit(-1.0).alias("Premium")
)

whole_data = pl.concat([train_data, test_data], how="vertical")

target_feature = "Premium"

categorical_features = [
    "PlanType",
    "PropertyType",
    "MaritalStatus",
    "JobRole",
    "Feedback",
    "ResidenceType",
    "Smoking",
    "Sex",
    "ExerciseFreq",
    "EducationLevel",
]

numeric_features = [
    col_name
    for col_name in whole_data.columns
    if col_name not in categorical_features
    and col_name != target_feature
    and col_name != "RowId"
]

ordinal_features = {
    "PlanType": ["Unknown", "basic", "comprehensive", "premium"],
    "Feedback": ["Unknown", "poor", "average", "good"],
    "ExerciseFreq": ["Unknown", "rarely", "monthly", "weekly", "daily"],
    "EducationLevel": ["Unknown", "high school", "bachelor's", "master's", "phd"],
}


nominal_features = [
    col_name
    for col_name in categorical_features
    if col_name not in ordinal_features.keys()
]


ordinal_pipe = Pipeline(
    steps=[
        (
            "categorical_imputer",
            SimpleImputer(strategy="constant", fill_value="Unknown"),
        ),
        (
            "ordinal_encoder",
            OrdinalEncoder(
                categories=[categories for categories in ordinal_features.values()],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)


nominal_pipe = Pipeline(
    steps=[
        (
            "categorical_imputer",
            SimpleImputer(strategy="constant", fill_value="Unknown"),
        ),
        ("onehot_encoder", OneHotEncoder(handle_unknown="warn", drop="first")),
    ]
)


skewed_numerical_features = ["EarningsBracket", "AnnualIncome"]

unskewed_numerical_features = [
    feature for feature in numeric_features if feature not in skewed_numerical_features
]

skewed_numeric_pipe = Pipeline(
    steps=[
        ("log1p", FunctionTransformer(lambda x: np.log1p(x))),
        ("simple_imputer", SimpleImputer(strategy="mean")),
        ("feature_normalizer", StandardScaler()),
    ]
)

unskewed_numeric_pipe = Pipeline(
    steps=[
        ("simple_imputer", SimpleImputer(strategy="mean")),
        ("feature_normalizer", StandardScaler()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal_categorical", ordinal_pipe, list(ordinal_features.keys())),
        ("nominal_categorical", nominal_pipe, nominal_features),
        ("skewed_numerical", skewed_numeric_pipe, skewed_numerical_features),
        ("unskewed_numerical", unskewed_numeric_pipe, unskewed_numerical_features),
    ],
    remainder="passthrough",
)

X_train, y_train = (
    train_data.drop("RowId", "Premium").to_pandas(),
    train_data.select(pl.col("Premium")).to_pandas(),
)


models_grid = {
    "random_forest": {
        "model": Pipeline(
            [
                ("preprocessoing", preprocessor),
                ("model", RandomForestRegressor(random_state=42, n_jobs=-1, verbose=3)),
            ]
        ),
        "parameters": {
            "model__n_estimators": scpt.randint(10, 100),
            "model__max_depth": scpt.randint(5, 30),
            "model__min_samples_split": scpt.randint(2, 15),
            "model__min_samples_leaf": scpt.randint(1, 15),
        },
    },
    "xgboost": {
        "model": Pipeline(
            [
                ("preprocessoing", preprocessor),
                ("model", xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=3)),
            ]
        ),
        "parameters": {
            "model__n_estimators": scpt.randint(10, 150),
            "model__max_depth": scpt.randint(10, 50),
            "model__learning_rate": scpt.uniform(0.4, 1),
            "model__subsample": scpt.uniform(0.4, 0.9),
            "model__colsample_bytree": scpt.uniform(0.2, 0.8),
        },
    },
    "lgbm_gdbt": {
        "model": Pipeline(
            [
                ("preprocessoing", preprocessor),
                (
                    "model",
                    lgb.LGBMRegressor(
                        boosting_type="gbdt",
                        random_state=42,
                        metric="rmse",
                        verbose=3,
                    ),
                ),
            ]
        ),
        "parameters": {
            "model__n_estimators": scpt.randint(10, 150),
            "model__max_depth": scpt.randint(5, 40),
            "model__learning_rate": scpt.uniform(0.3, 1),
            "model__subsample": scpt.uniform(0.4, 0.9),
            "model__colsample_bytree": scpt.uniform(0.2, 0.8),
            "model__num_leaves": scpt.randint(20, 50),
            "model__min_child_samples": scpt.randint(10, 30),
        },
    },
    "lgbm_goss": {
        "model": Pipeline(
            [
                ("preprocessoing", preprocessor),
                (
                    "model",
                    lgb.LGBMRegressor(
                        boosting_type="goss",
                        random_state=42,
                        metric="rmse",
                        verbose=3,
                    ),
                ),
            ]
        ),
        "parameters": {
            "model__n_estimators": scpt.randint(10, 150),
            "model__max_depth": scpt.randint(5, 40),
            "model__learning_rate": scpt.uniform(0.3, 1),
            "model__subsample": scpt.uniform(0.4, 0.9),
            "model__colsample_bytree": scpt.uniform(0.2, 0.8),
            "model__num_leaves": scpt.randint(20, 50),
            "model__min_child_samples": scpt.randint(10, 30),
        },
    },
}

for model_type, model_params in models_grid.items():
    print("-" * 10, model_type, "-" * 10)
    model = model_params["model"]
    parameters = model_params["parameters"]
    random_grid = RandomizedSearchCV(
        model,
        parameters,
        n_iter=30,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=3,
        random_state=42,
    )

    random_grid.fit(X_train, y_train.to_numpy().ravel())

    print(random_grid.best_params_)
    print(random_grid.best_score_)
    print(random_grid.best_estimator_)
