import pickle
from pathlib import Path

import click
import mlflow
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed and serialized NYC taxi trip data was saved",
)
def evaluate_champion_model(data_path: str | Path = Path("./output")):

    # Load data
    if isinstance(data_path, str):
        data_path = Path(data_path)
    with open(data_path / "test.pkl", "rb") as test_serialized:
        X_test, y_test = pickle.load(test_serialized)

    # Obtain the version aliased "champion" for the model "best-model-homework" from the model registry
    client = mlflow.MlflowClient()
    champion_version = client.get_model_version_by_alias(
        name="best-model-homework", alias="champion"
    )

    # Load the model and use it to make a prediction
    champion_model = mlflow.sklearn.load_model(f"runs:/{champion_version.run_id}/model")
    y_pred = champion_model.predict(X_test)

    # Compute metric and print it
    print(
        "Champion model evaluation RMSE: {:.5f}".format(
            root_mean_squared_error(y_pred, y_test)
        )
    )


if __name__ == "__main__":
    evaluate_champion_model()
