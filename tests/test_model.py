import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.predict import DEFAULT_WINE_VALUES, load_latest_model, predict_wine_quality


@pytest.fixture
def mock_extracted_data():
    """Provides a complete sample feature dictionary mirroring LLM output."""
    return {
        "wine_type": "red",
        "fixed acidity": 7.4,
        "volatile acidity": 0.70,
        "citric acid": 0.00,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }


def test_prediction_output_shape_and_type(mock_extracted_data):
    """6.2.1: Verify model inference returns a single float prediction score."""
    model, config, _ = load_latest_model()
    score, cleaned_features, missing_imputed = predict_wine_quality(
        mock_extracted_data, model, config
    )

    # Assert score is a valid scalar float between reasonable quality bounds (1-10)
    assert isinstance(score, float)
    assert 1.0 <= score <= 10.0
    assert isinstance(cleaned_features, dict)
    assert isinstance(missing_imputed, list)


def test_model_performance_threshold():
    """6.2.2: Verify champion model maintains an acceptable baseline RMSE threshold."""
    model, config, _ = load_latest_model()

    # Benchmark samples with varying quality characteristics
    benchmark_samples = [
        DEFAULT_WINE_VALUES.copy(),
        {**DEFAULT_WINE_VALUES, "alcohol": 11.5, "volatile acidity": 0.3},
        {**DEFAULT_WINE_VALUES, "alcohol": 9.0, "volatile acidity": 0.8},
        {**DEFAULT_WINE_VALUES, "citric acid": 0.45, "residual sugar": 8.0},
        {**DEFAULT_WINE_VALUES, "pH": 3.10, "sulphates": 0.75},
    ]
    benchmark_targets = np.array([5.5, 6.5, 4.5, 6.0, 6.2])

    # Run inference via predict_wine_quality to apply encoding & preprocessing
    predictions = []
    for sample in benchmark_samples:
        score, _, _ = predict_wine_quality(sample, model, config)
        predictions.append(score)

    predictions = np.array(predictions)
    rmse = np.sqrt(np.mean((predictions - benchmark_targets) ** 2))

    # Baseline sanity check: RMSE should be less than 3.0 on standard scale
    assert rmse < 3.0