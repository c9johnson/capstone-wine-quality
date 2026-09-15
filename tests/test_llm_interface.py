import json
from unittest.mock import MagicMock
import pytest
from src.llm_interface import extract_features_from_prompt


@pytest.fixture
def mock_llm_json_response():
    """Provides a valid JSON payload simulating LLM API output."""
    return json.dumps(
        {
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
    )


def test_extract_features_from_prompt_success(mock_llm_json_response):
    """6.3.1: Verify LLM feature extraction parses valid JSON responses accurately."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[
        0
    ].message.content = mock_llm_json_response

    sample_prompt = "Red wine with 7.4 fixed acidity, pH of 3.51, and 9.4% alcohol."

    extracted = extract_features_from_prompt(sample_prompt, client=mock_client)

    assert isinstance(extracted, dict)
    assert extracted.get("wine_type") == "red"
    assert extracted.get("alcohol") == 9.4
    assert extracted.get("pH") == 3.51


def test_extract_features_from_prompt_failure():
    """6.3.2: Verify extraction handles API exceptions gracefully."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception(
        "API Connection Error"
    )

    extracted = extract_features_from_prompt("Some prompt", client=mock_client)

    # Adjust assertion based on your error handling (returns empty dict or None)
    assert extracted == {} or extracted is None