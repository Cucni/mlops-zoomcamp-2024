import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import scipy.sparse
import xgboost as xgb
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

DATA_FOLDER = Path("../../data")
MODEL_FOLDER = Path("../models")
MLFLOW_DB_PATH = Path("..")


@task(retries=3, retry_delay_seconds=0.1)
def read_dataframe(filename: Path) -> pd.DataFrame:
    categorical = ["PULocationID", "DOLocationID"]

    df = pd.read_parquet(filename)

    df["duration"] = (
        df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[categorical] = df[categorical].astype(str)

    return df


@task
def add_interaction_feature(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple[
    scipy.sparse._csr.csr_matrix,
    scipy.sparse._csr.csr_matrix,
    pd.Series,
    pd.Series,
    DictVectorizer,
]:
    # Engineer an interaction feature
    # Interaction features combine two (or more) features into one. The idea is that the pair of values matters more than the single values together.
    # In our case we will create an interaction feature from PU and DO. The interpretation is that instead of capturing what is the marginal variation due to "pick up here" and "drop off there" separately, we'll try to capture the information carried by "pick up here and drop off there".
    # Note that this also reduces in half the number of features, reducing the variance of the models.

    # In this case it suffices to combine the string to create a new categorical feature
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    # Use DictVectorizer to encode categorical features and obtain training and validation matrices
    # When we use the interaction features, we drop the individual features.
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    # Obtain training and validation targets
    target = "duration"
    y_train = df_train[target]
    y_val = df_val[target]

    return X_train, X_val, y_train, y_val, dv


@task(log_prints=True)
def train_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    dv: DictVectorizer,
) -> None:
    with mlflow.start_run():
        # Training of the best model as inferred from the mlflow ui, by choosing the model that minimizes the RMSE loss
        params = {
            "learning_rate": 0.6424019772458974,
            "max_depth": 20,
            "min_child_weight": 2.2694144028711833,
            "objective": "reg:linear",
            "reg_alpha": 0.025551415216516424,
            "reg_lambda": 0.009147735459264332,
            "seed": 42,
        }

        # we use xgb's internal DMatrix to store the data for trainig and cross-validation
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=15,
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Logging it as a model using the mlflow API for the model framework (xgboost)
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        # Logging the preprocessor as an artifact
        # First we save it as pickle, and then we log it by referencing the path
        MODEL_FOLDER.mkdir(exist_ok=True)
        with open(MODEL_FOLDER / "preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(
            MODEL_FOLDER / "preprocessor.b", artifact_path="preprocessor"
        )


@flow
def main_flow(
    train_path: Path = Path(DATA_FOLDER / "green_tripdata_2021-01.parquet"),
    val_path: Path = Path(DATA_FOLDER / "green_tripdata_2021-02.parquet"),
) -> None:
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}/mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    X_train, X_val, y_train, y_val, dv = add_interaction_feature(df_train, df_val)

    train_model(X_train, X_val, y_train, y_val, dv)


if __name__ == "__main__":
    main_flow()
