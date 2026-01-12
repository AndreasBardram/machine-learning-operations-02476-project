from pathlib import Path

import lightning as pl
import torch
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from lightning import seed_everything
from torch.utils.data import DataLoader, random_split

from ml_ops_project.data import PROCESSED_NAME, MyDataset
from ml_ops_project.model import TransactionModel

torch.set_float32_matmul_precision("high")


def train(subset: bool = False):
    pl.seed_everything(42)  # Set seed for reproducibility

    # 1. Configuration
    batch_size = 64
    max_epochs = 100
    if subset:
        data_path = Path("data/processed") / f"{PROCESSED_NAME}_subset"
    else:
        data_path = Path("data/processed") / PROCESSED_NAME

    # 2. Data
    print(f"Loading data from {data_path}...")
    full_dataset = MyDataset(data_path)
    full_dataset.load()

    print("Starting random split...")
    # Split 80/10/10 Simple
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    print(f"Splitting dataset: Train={train_size}, Val={val_size}, Test={test_size}")

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True)

    # 3. Model
    # We know input_dim=32 and output_dim=10 from preprocessing
    model = TransactionModel(
        input_dim=32,
        hidden_dim=1024,
        output_dim=10,
        num_layers=5,
        dropout_p=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        l1_lambda=1e-5,
    )

    # 4. Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/checkpoints",
        filename="transaction-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Profiler
    profiler = AdvancedProfiler(dirpath="logs/profiler", filename="perf_logs_advanced")
    # profiler = SimpleProfiler(dirpath="logs/profiler", filename="perf_logs_simple")

    # 5. Trainer
    wandb_logger = WandbLogger(project="ml_ops_project", name="transaction_model_run")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        logger=wandb_logger,
        accelerator="auto",  # Use GPU if available
        devices="auto",
        # profiler=profiler,
    )

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Testing best model...")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    typer.run(train)
