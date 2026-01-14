import sys
from pathlib import Path

import pytest
import torch

# Add root path to dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ml_ops_project.model import TransactionModel  # noqa: E402


@pytest.fixture
def model():
    return TransactionModel(input_dim=32, hidden_dim=64, output_dim=10)


@pytest.fixture
def batch():
    batch_size = 8
    return {"features": torch.randn(batch_size, 32), "labels": torch.randint(0, 10, (batch_size,))}


def test_forward_shape(model):
    x = torch.randn(4, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_forward_dtype(model):
    x = torch.randn(2, 32)
    out = model(x)
    assert out.dtype == torch.float32


def test_training_step_returns_loss(model, batch):
    loss = model.training_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar loss


def test_validation_step_returns_loss(model, batch):
    loss = model.validation_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_test_step_returns_loss(model, batch):
    loss = model.test_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_loss_decreases_with_perfect_prediction(model):
    logits = torch.zeros(4, 10)
    labels = torch.tensor([0, 1, 2, 3])
    logits[range(4), labels] = 10.0  # force correct predictions

    loss = model.criterion(logits, labels)
    assert loss.item() < 1e-2


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == model.learning_rate
