import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval_metrics(actual, pred) -> dict:
    """Calculate regression performance metrics."""
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, pred)

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
    return metrics


if __name__ == "__main__":
    # Quick sanity test on dummy inputs
    y_true = [3.0, 5.0, 2.5, 7.0]
    y_pred = [2.5, 5.1, 2.4, 6.8]
    test_metrics = eval_metrics(y_true, y_pred)
    print("Sanity check metrics:", test_metrics)