import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration parameters from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer using feature specs from config."""
    num_cols = config["features"]["numerical"]
    cat_cols = config["features"]["categorical"]
    impute_strategy = config["preprocessing"]["impute_strategy"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )
    return preprocessor


def preprocess_data(config_path: str = "configs/config.yaml"):
    """Load data, perform train/test split, and transform features using YAML config."""
    config = load_config(config_path)

    # Read raw data using path specified in config
    raw_path = config["data"]["raw_path"]
    df = pd.read_csv(raw_path)

    target_col = config["data"]["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    preprocessor = build_preprocessor(config)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, preprocessor = preprocess_data()
    print("Preprocessed successfully using configs/config.yaml!")
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")