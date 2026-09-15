import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from src.preprocess import load_config, preprocess_data

# Matches exact feature list from data/winequality.csv (excluding target 'quality')
DEFAULT_WINE_VALUES = {
    "fixed acidity": 7.0,
    "volatile acidity": 0.3,
    "citric acid": 0.3,
    "residual sugar": 2.5,
    "chlorides": 0.08,
    "free sulfur dioxide": 30.0,
    "total sulfur dioxide": 115.0,
    "density": 0.995,
    "pH": 3.2,
    "sulphates": 0.5,
    "alcohol": 10.5,
    "wine_type": "white"
}

FEATURE_ORDER = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type"
]

def load_latest_model():
    """
    Programmatically queries MLflow tracking using search_runs() to find the 
    champion model based on lowest RMSE.
    """
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "config.yaml"
    
    tracking_uri = "sqlite:///mlflow.db"
    if config_path.exists():
        try:
            config = load_config(str(config_path))
            tracking_uri = config.get("mlflow", {}).get("tracking_uri", tracking_uri)
        except Exception:
            pass

    mlflow.set_tracking_uri(tracking_uri)

    try:
        runs = mlflow.search_runs(
            order_by=["metrics.rmse ASC"],
            max_results=1
        )
    except Exception:
        runs = pd.DataFrame()

    if not runs.empty:
        best_run = runs.iloc[0]
        best_run_id = best_run.run_id
        model_uri = f"runs:/{best_run_id}/model"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            
            model_type_str = "Random Forest Regressor"
            if "params.model_type" in best_run and pd.notna(best_run["params.model_type"]):
                model_type_str = str(best_run["params.model_type"]).replace("_", " ").title()
                
            config_meta = {"model": {"type": model_type_str}}
            return model, config_meta, best_run_id
        except Exception:
            pass

    mlruns_path = project_root / "mlruns"
    if mlruns_path.exists():
        found_models = list(mlruns_path.glob("**/model.pkl"))
        if found_models:
            model_path = max(found_models, key=lambda p: p.stat().st_mtime)
            model = joblib.load(model_path)
            best_run_id = model_path.parts[-3] if len(model_path.parts) >= 3 else "local_run"
            config_meta = {"model": {"type": "Random Forest Regressor"}}
            return model, config_meta, best_run_id

    raise FileNotFoundError(f"No MLflow model runs found at tracking URI: {tracking_uri}. Run train.py first.")

def predict_wine_quality(extracted_data: dict, model, config):
    """
    Validates features, imputes missing values, orders columns to match training set,
    applies preprocessing transformation, and executes model prediction.
    """
    features_clean = {}
    missing_imputed = []
    
    # Process each feature according to expected type and order
    for feature in FEATURE_ORDER:
        val = extracted_data.get(feature)
        default_val = DEFAULT_WINE_VALUES[feature]
        
        if feature == "wine_type":
            if val and str(val).lower() in ["white", "red"]:
                features_clean[feature] = str(val).lower()
            else:
                features_clean[feature] = default_val
                missing_imputed.append(feature)
        else:
            if val is None:
                features_clean[feature] = default_val
                missing_imputed.append(feature)
            else:
                try:
                    features_clean[feature] = float(val)
                except (ValueError, TypeError):
                    features_clean[feature] = default_val
                    missing_imputed.append(feature)

    # Construct DataFrame with exact column order expected by training pipeline
    input_df = pd.DataFrame([features_clean])[FEATURE_ORDER]
    
    project_root = Path(__file__).resolve().parent.parent
    preprocessor_path = project_root / "models" / "preprocessor.pkl"
    
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        processed_input = preprocessor.transform(input_df)
    else:
        try:
            config_path = str(project_root / "configs" / "config.yaml")
            _, _, _, _, preprocessor = preprocess_data(config_path)
            processed_input = preprocessor.transform(input_df)
        except Exception as e:
            raise RuntimeError(f"Preprocessing transformation failed during inference: {e}")

    prediction = model.predict(processed_input)[0]
    score = float(prediction)

    return score, features_clean, missing_imputed