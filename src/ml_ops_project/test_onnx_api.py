import pytest
import requests

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def api_url():
    """Fixture to provide the base API URL."""
    return BASE_URL


@pytest.fixture(scope="module")
def sample_features():
    """Fixture to provide sample transaction features."""
    return [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        0.1,
        0.2,
    ]


@pytest.fixture(scope="module")
def sample_batch_transactions():
    """Fixture to provide sample batch transactions."""
    return [[0.1] * 32, [0.2] * 32, [0.3] * 32]


class TestAPIHealth:
    """Test suite for API health and info endpoints."""

    def test_root_endpoint(self, api_url):
        """Test that the root endpoint returns expected information."""
        response = requests.get(f"{api_url}/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert "/predict" in data["endpoints"]

    def test_health_check(self, api_url):
        """Test that the health check endpoint returns healthy status."""
        response = requests.get(f"{api_url}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "providers" in data
        assert len(data["providers"]) > 0

    def test_model_info(self, api_url):
        """Test that model info endpoint returns correct model details."""
        response = requests.get(f"{api_url}/model_info")

        assert response.status_code == 200
        data = response.json()
        assert "input_name" in data
        assert "input_shape" in data
        assert "output_name" in data
        assert "output_shape" in data
        assert "providers" in data


class TestPredictionEndpoint:
    """Test suite for single prediction endpoint."""

    def test_predict_success(self, api_url, sample_features):
        """Test that prediction endpoint returns valid prediction."""
        response = requests.post(f"{api_url}/predict", json={"features": sample_features})

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "predicted_class" in data
        assert "probabilities" in data
        assert "confidence" in data

        # Check data types and ranges
        assert isinstance(data["predicted_class"], int)
        assert 0 <= data["predicted_class"] <= 9
        assert isinstance(data["probabilities"], list)
        assert len(data["probabilities"]) == 10
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

        # Check probabilities sum to 1
        prob_sum = sum(data["probabilities"])
        assert abs(prob_sum - 1.0) < 1e-5

        # Check confidence matches max probability
        max_prob = max(data["probabilities"])
        assert abs(data["confidence"] - max_prob) < 1e-5

    def test_predict_invalid_feature_count_too_few(self, api_url):
        """Test that prediction fails with too few features."""
        response = requests.post(
            f"{api_url}/predict",
            json={"features": [0.1] * 20},  # Only 20 features instead of 32
        )

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_feature_count_too_many(self, api_url):
        """Test that prediction fails with too many features."""
        response = requests.post(
            f"{api_url}/predict",
            json={"features": [0.1] * 40},  # 40 features instead of 32
        )

        assert response.status_code == 422  # Validation error

    def test_predict_missing_features(self, api_url):
        """Test that prediction fails when features are missing."""
        response = requests.post(f"{api_url}/predict", json={})

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_feature_types(self, api_url):
        """Test that prediction handles invalid feature types."""
        response = requests.post(f"{api_url}/predict", json={"features": ["invalid"] * 32})

        assert response.status_code == 422  # Validation error


class TestBatchPredictionEndpoint:
    """Test suite for batch prediction endpoint."""

    def test_predict_batch_success(self, api_url, sample_batch_transactions):
        """Test that batch prediction returns valid predictions."""
        response = requests.post(f"{api_url}/predict_batch", json={"transactions": sample_batch_transactions})

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "predictions" in data
        assert "probabilities" in data
        assert "confidences" in data

        # Check lengths match number of transactions
        num_transactions = len(sample_batch_transactions)
        assert len(data["predictions"]) == num_transactions
        assert len(data["probabilities"]) == num_transactions
        assert len(data["confidences"]) == num_transactions

        # Check each prediction
        for i in range(num_transactions):
            assert isinstance(data["predictions"][i], int)
            assert 0 <= data["predictions"][i] <= 9

            assert isinstance(data["probabilities"][i], list)
            assert len(data["probabilities"][i]) == 10

            assert isinstance(data["confidences"][i], float)
            assert 0.0 <= data["confidences"][i] <= 1.0

            # Check probabilities sum to 1
            prob_sum = sum(data["probabilities"][i])
            assert abs(prob_sum - 1.0) < 1e-5

    def test_predict_batch_empty(self, api_url):
        """Test batch prediction with empty transaction list."""
        response = requests.post(f"{api_url}/predict_batch", json={"transactions": []})

        assert response.status_code == 500  # Internal server error due to no data

    def test_predict_batch_invalid_feature_count(self, api_url):
        """Test that batch prediction fails with invalid feature count."""
        invalid_transactions = [
            [0.1] * 32,
            [0.2] * 20,  # Invalid: only 20 features
            [0.3] * 32,
        ]
        response = requests.post(f"{api_url}/predict_batch", json={"transactions": invalid_transactions})

        assert response.status_code == 400  # Bad request
        data = response.json()
        assert "detail" in data
        assert "Transaction 1" in data["detail"]

    def test_predict_batch_single_transaction(self, api_url, sample_features):
        """Test batch prediction with single transaction."""
        response = requests.post(f"{api_url}/predict_batch", json={"transactions": [sample_features]})

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        assert len(data["probabilities"]) == 1
        assert len(data["confidences"]) == 1


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_predict_with_zeros(self, api_url):
        """Test prediction with all zero features."""
        response = requests.post(f"{api_url}/predict", json={"features": [0.0] * 32})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data

    def test_predict_with_negative_values(self, api_url):
        """Test prediction with negative feature values."""
        response = requests.post(f"{api_url}/predict", json={"features": [-0.5] * 32})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data

    def test_predict_with_large_values(self, api_url):
        """Test prediction with large feature values."""
        response = requests.post(f"{api_url}/predict", json={"features": [1000.0] * 32})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data

    def test_invalid_endpoint(self, api_url):
        """Test that invalid endpoints return 404."""
        response = requests.get(f"{api_url}/invalid_endpoint")

        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
