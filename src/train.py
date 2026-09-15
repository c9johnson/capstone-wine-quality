import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from preprocess import preprocess_data, load_config
from evaluate import eval_metrics


def get_model(model_type: str, params: dict):
    """Factory function to instantiate scikit-learn models dynamically."""
    if model_type == "random_forest":
        return RandomForestRegressor(**params)
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    elif model_type == "ridge":
        return Ridge(**params)
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'")


def train_model(config_path: str = "configs/config.yaml"):
    """Train a regression model and log parameters, metrics, and artifacts to MLflow."""
    config = load_config(config_path)

    # MLflow Setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(config_path)

    model_type = config["model"]["type"]
    params = config["model"]["params"]

    # Start MLflow Run
    with mlflow.start_run():
        # Instantiate model based on config setting
        model = get_model(model_type, params)

        # Train model
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        metrics = eval_metrics(y_test, predictions)

        # Log parameters, metrics, and model artifact
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics(metrics)
        
        # Updated to 'name' to resolve deprecation warning
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

        print(f"Run logged successfully for [{model_type}]!")
        print(f"Metrics - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    train_model()