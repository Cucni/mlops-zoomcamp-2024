```
# Start an mlflow server with the default configurations and open the web UI
mlflow
mlflow ui

# Start an mlflow server by specifying the backend store URI. When using a database for this, we are also using a model registry.
mlflow server --backend-store-uri sqlite:///mlflow.db
mlflow ui

# Start an mlflow server by specifying a database as the backend store URI and a destination where to store the artifacts.
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination artifacts
mlflow ui
```