import numpy as np
import onnxruntime as rt
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="Transaction Classification API",
    description="API for classifying transactions using ONNX model",
    version="1.0.0",
)

# Load the ONNX model at startup
provider_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ort_session = rt.InferenceSession("models/transaction_model.onnx", providers=provider_list)

# Get input and output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name


# Define request schema
class TransactionFeatures(BaseModel):
    features: list[float] = Field(..., description="List of 32 transaction features", min_length=32, max_length=32)

    class Config:
        json_schema_extra = {  # noqa: RUF012
            "example": {
                "features": [0.1] * 32  # Example with 32 features
            }
        }


# Define response schema
class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="Predicted transaction category (0-9)")
    probabilities: list[float] = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Confidence of the prediction (max probability)")


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Transaction Classification API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/predict_batch": "POST - Make batch predictions",
            "/health": "GET - Check API health",
            "/model_info": "GET - Get model information",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": ort_session is not None, "providers": ort_session.get_providers()}


@app.get("/model_info")
def model_info():
    """Get information about the loaded model."""
    input_info = ort_session.get_inputs()[0]
    output_info = ort_session.get_outputs()[0]

    return {
        "input_name": input_info.name,
        "input_shape": input_info.shape,
        "input_type": input_info.type,
        "output_name": output_info.name,
        "output_shape": output_info.shape,
        "output_type": output_info.type,
        "providers": ort_session.get_providers(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    """
    Predict transaction category from features.

    Args:
        transaction: TransactionFeatures object with 32 features

    Returns:
        PredictionResponse with predicted class, probabilities, and confidence
    """
    try:
        # Convert input to numpy array with correct shape (1, 32)
        input_data = np.array([transaction.features], dtype=np.float32)

        # Run inference
        outputs = ort_session.run([output_name], {input_name: input_data})
        logits = outputs[0][0]  # Get the first (and only) prediction

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Get predicted class and confidence
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        return PredictionResponse(
            predicted_class=predicted_class, probabilities=probabilities.tolist(), confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}")  # noqa: B904


# Batch prediction schema
class BatchTransactionFeatures(BaseModel):
    transactions: list[list[float]] = Field(
        ..., description="List of transaction feature vectors (each with 32 features)"
    )

    class Config:
        json_schema_extra = {"example": {"transactions": [[0.1] * 32, [0.2] * 32, [0.3] * 32]}}  # noqa: RUF012


class BatchPredictionResponse(BaseModel):
    predictions: list[int] = Field(..., description="Predicted classes for each transaction")
    probabilities: list[list[float]] = Field(..., description="Probabilities for each transaction")
    confidences: list[float] = Field(..., description="Confidence scores for each prediction")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchTransactionFeatures):
    """
    Predict transaction categories for multiple transactions at once.

    Args:
        batch: BatchTransactionFeatures with multiple transaction feature vectors

    Returns:
        BatchPredictionResponse with predictions for all transactions
    """
    try:
        # Validate that all transactions have 32 features
        for i, transaction in enumerate(batch.transactions):
            if len(transaction) != 32:
                raise HTTPException(
                    status_code=400, detail=f"Transaction {i} has {len(transaction)} features, expected 32"
                )

        # Convert to numpy array
        input_data = np.array(batch.transactions, dtype=np.float32)

        # Run inference
        outputs = ort_session.run([output_name], {input_name: input_data})
        logits = outputs[0]

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Get predictions and confidences
        predictions = np.argmax(probabilities, axis=1).tolist()
        confidences = np.max(probabilities, axis=1).tolist()

        return BatchPredictionResponse(
            predictions=predictions, probabilities=probabilities.tolist(), confidences=confidences
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e!s}")  # noqa: B904


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
