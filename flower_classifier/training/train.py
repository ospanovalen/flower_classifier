import os
from pathlib import Path
from typing import Dict, Tuple

import hydra
import lightning as L
import mlflow
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from flower_classifier.data.dataset import get_data_loaders, get_default_transforms
from flower_classifier.models.flower_model import FlowerClassifier
from flower_classifier.utils import init_basic_logger


def setup_directories(cfg: DictConfig) -> None:
    """Create necessary directories for saving models and plots."""
    Path(cfg.paths.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.plots_dir).mkdir(parents=True, exist_ok=True)


def setup_mlflow(cfg: DictConfig) -> MLFlowLogger:
    """Setup MLflow logger for experiment tracking."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    try:
        experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
        if experiment is None:
            mlflow.create_experiment(cfg.mlflow.experiment_name)
    except Exception as e:
        logger = init_basic_logger(__name__)
        logger.warning(f"Could not setup MLflow experiment: {e}")

    return MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=cfg.run_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )


def create_data_loaders(cfg: DictConfig) -> Tuple[L.LightningDataModule, Dict]:
    """Create data loaders for training, validation, and testing."""
    # Set random seed for reproducibility
    torch.manual_seed(cfg.data.seed)

    # Create transforms
    transforms = get_default_transforms(size=cfg.data.image_size)

    # Create data loaders
    train_loader, val_loader, test_loader, class_names, class_counts = get_data_loaders(
        root=cfg.paths.data_dir,
        transformations=transforms,
        batch_size=cfg.data.batch_size,
        split=[cfg.data.split.train, cfg.data.split.val, cfg.data.split.test],
        num_workers=cfg.data.num_workers,
    )

    # Create a simple data module
    class FlowerDataModule(L.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

        def train_dataloader(self):
            return self.train_loader

        def val_dataloader(self):
            return self.val_loader

        def test_dataloader(self):
            return self.test_loader

    return FlowerDataModule(), {
        "class_names": class_names,
        "class_counts": class_counts,
    }


def create_model(cfg: DictConfig) -> FlowerClassifier:
    """Create the flower classification model."""
    return FlowerClassifier(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        contrastive_margin=cfg.model.contrastive_margin,
    )


def create_callbacks(cfg: DictConfig) -> list:
    """Create Lightning callbacks for training."""
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.save_dir,
        filename=cfg.callbacks.model_checkpoint.filename,
        monitor=cfg.callbacks.model_checkpoint.monitor,
        mode=cfg.callbacks.model_checkpoint.mode,
        save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
        save_last=cfg.callbacks.model_checkpoint.save_last,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor=cfg.callbacks.early_stopping.monitor,
        mode=cfg.callbacks.early_stopping.mode,
        patience=cfg.callbacks.early_stopping.patience,
        verbose=cfg.callbacks.early_stopping.verbose,
    )
    callbacks.append(early_stopping)

    return callbacks


def train_model(cfg: DictConfig) -> None:
    """Main training function."""
    logger = init_basic_logger(__name__)
    logger.info("Starting flower classification training...")

    # Setup directories
    setup_directories(cfg)

    # Setup MLflow
    mlf_logger = setup_mlflow(cfg)

    # Create data loaders
    data_module, data_info = create_data_loaders(cfg)
    logger.info(f"Dataset info: {data_info}")

    # Create model
    model = create_model(cfg)

    # Create callbacks
    callbacks = create_callbacks(cfg)

    # Create trainer
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=mlf_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    logger.info("Training completed!")
    logger.info(f"Best model saved to: {cfg.paths.save_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    train_model(cfg)


if __name__ == "__main__":
    main()
