import mlflow
import pandas as pd
from src.preprocess import load_config

def compare_runs(config_path: str = "configs/config.yaml"):
    """Query MLflow tracking database using mlflow.search_runs() and rank experiments."""
    config = load_config(config_path)
    
    # Connect to SQLite tracking database
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    
    # Query all completed runs for this experiment
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'"
    )
    
    if runs.empty:
        print("No completed MLflow runs found.")
        return

    # Extract key metrics and hyperparameter columns for clean display
    display_cols = [
        "run_id",
        "params.model_type",
        "metrics.r2",
        "metrics.rmse",
        "metrics.mae",
        "start_time"
    ]
    
    # Filter to columns that exist in the query result
    available_cols = [c for c in display_cols if c in runs.columns]
    summary_df = runs[available_cols].copy()
    
    # Sort runs by highest R2 score
    if "metrics.r2" in summary_df.columns:
        summary_df = summary_df.sort_values(by="metrics.r2", ascending=False)

    print("\n================ MLFLOW EXPERIMENT COMPARISON ================")
    print(summary_df.to_string(index=False))
    
    # Identify the best run
    best_run = summary_df.iloc[0]
    print("\n================ WINNING RUN SELECTION ================")
    print(f"Best Run ID : {best_run['run_id']}")
    print(f"Model Type  : {best_run.get('params.model_type', 'N/A')}")
    print(f"Best R2     : {best_run.get('metrics.r2', 'N/A'):.4f}")
    print(f"Best RMSE   : {best_run.get('metrics.rmse', 'N/A'):.4f}")


if __name__ == "__main__":
    compare_runs()