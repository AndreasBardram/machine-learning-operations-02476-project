import numpy as np
import onnxruntime as rt

provider_list = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# Load the ONNX model with specified providers
ort_session = rt.InferenceSession("models/transaction_model.onnx", providers=provider_list)

# Get input and output names
input_names = [i.name for i in ort_session.get_inputs()]
output_names = [i.name for i in ort_session.get_outputs()]

print(f"Input names: {input_names}")
print(f"Output names: {output_names}")

# Get input shape information
for i in ort_session.get_inputs():
    print(f"Input '{i.name}' shape: {i.shape}, dtype: {i.type}")

# Create dummy input with correct shape (batch_size=1, input_dim=32)
dummy_input = np.random.randn(1, 32).astype(np.float32)

# Create batch dictionary
batch = {input_names[0]: dummy_input}

# Run inference
output = ort_session.run(output_names, batch)

print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output[0].shape}")
print(f"Output logits: {output[0]}")
print(f"Predicted class: {np.argmax(output[0], axis=1)[0]}")

# Apply softmax to get probabilities
probabilities = np.exp(output[0]) / np.sum(np.exp(output[0]), axis=1, keepdims=True)
print(f"Probabilities: {probabilities}")
