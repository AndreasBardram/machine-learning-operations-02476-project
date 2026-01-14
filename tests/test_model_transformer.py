import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Add root path to dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.ml_ops_project.model_transformer import TransformerTransactionModel  # noqa: E402


class DummyHFModel(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.classifier = torch.nn.Linear(16, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: ARG002
        batch_size = input_ids.size(0)
        hidden = torch.randn(batch_size, 16)
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return SimpleNamespace(loss=loss, logits=logits)


@pytest.fixture
def model(monkeypatch):
    def mock_from_pretrained(*args, **kwargs):  # noqa: ARG001
        return DummyHFModel(num_labels=kwargs["num_labels"])

    monkeypatch.setattr(
        "src.ml_ops_project.model_transformer.AutoModelForSequenceClassification.from_pretrained",
        mock_from_pretrained,
    )

    return TransformerTransactionModel(num_labels=10)


@pytest.fixture
def batch():
    batch_size = 4
    seq_len = 8
    return {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, 10, (batch_size,)),
    }


def test_forward_outputs(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    assert hasattr(outputs, "loss")
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape == (4, 10)


def test_training_step_returns_loss(model, batch):
    loss = model.training_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_validation_step_returns_loss(model, batch):
    loss = model.validation_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_test_step_returns_loss(model, batch):
    loss = model.test_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_accuracy_computation():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean()
    assert acc.item() == 1.0


def test_configure_optimizers(model):
    optimizer = model.configure_optimizers()
    assert optimizer.__class__.__name__ == "AdamW"
    assert optimizer.defaults["lr"] == model.hparams.learning_rate
