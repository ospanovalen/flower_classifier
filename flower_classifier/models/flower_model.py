from typing import Dict, List

import lightning as L
import timm
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score, MulticlassStatScores


class FlowerClassifier(L.LightningModule):
    """Lightning module for flower classification with contrastive learning."""

    def __init__(
        self,
        model_name: str = "rexnet_150",
        num_classes: int = 5,
        learning_rate: float = 0.001,
        contrastive_margin: float = 0.3,
        **kwargs,
    ):
        """
        Initialize the flower classifier.

        Args:
            model_name: Name of the timm model to use
            num_classes: Number of flower classes
            learning_rate: Learning rate for optimization
            contrastive_margin: Margin for contrastive loss
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.contrastive_margin = contrastive_margin

        # Initialize model
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )

        # Loss functions
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.cs_loss_fn = nn.CosineEmbeddingLoss(margin=contrastive_margin)

        # Metrics
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average="micro")
        self.stat_scores = MulticlassStatScores(
            num_classes=num_classes, average="micro"
        )

        # Contrastive labels
        self.register_buffer("cos_pos", torch.tensor(1.0).unsqueeze(0))
        self.register_buffer("cos_neg", torch.tensor(-1.0).unsqueeze(0))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone model."""
        return self.backbone.forward_features(x)

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        """Apply classification head to features."""
        return self.backbone.forward_head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete model."""
        return self.backbone(x)

    def get_feature_maps(
        self, feature_tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply global average pooling to feature maps.

        Args:
            feature_tensors: List of feature tensors

        Returns:
            List of pooled feature tensors
        """
        if not feature_tensors:
            return []

        # Assume all tensors have the same spatial dimensions
        h, w = feature_tensors[0].shape[2], feature_tensors[0].shape[3]
        pool = nn.AvgPool2d((h, w))

        return [torch.reshape(pool(ft), (-1, ft.shape[1])) for ft in feature_tensors]

    def compute_contrastive_loss(
        self,
        qry_features: torch.Tensor,
        pos_features: torch.Tensor,
        neg_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between query, positive, and negative features.

        Args:
            qry_features: Query image features
            pos_features: Positive image features
            neg_features: Negative image features

        Returns:
            Contrastive loss value
        """
        pos_loss = self.cs_loss_fn(
            qry_features, pos_features, self.cos_pos.expand(qry_features.size(0))
        )
        neg_loss = self.cs_loss_fn(
            qry_features, neg_features, self.cos_neg.expand(qry_features.size(0))
        )
        return pos_loss + neg_loss

    def compute_classification_loss(
        self, qry_preds: torch.Tensor, pos_preds: torch.Tensor, qry_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss for query and positive predictions.

        Args:
            qry_preds: Query predictions
            pos_preds: Positive predictions
            qry_labels: True labels

        Returns:
            Classification loss value
        """
        qry_loss = self.ce_loss_fn(qry_preds, qry_labels)
        pos_loss = self.ce_loss_fn(pos_preds, qry_labels)
        return qry_loss + pos_loss

    def shared_step(
        self, batch: Dict[str, torch.Tensor], stage: str
    ) -> Dict[str, torch.Tensor]:
        """
        Shared step for training/validation/test.

        Args:
            batch: Batch of data
            stage: Stage name (train/val/test)

        Returns:
            Dictionary with loss and predictions
        """
        qry_ims = batch["qry_im"]
        pos_ims = batch["pos_im"]
        neg_ims = batch["neg_im"]
        qry_labels = batch["qry_gt"]

        # Extract features
        qry_features = self.forward_features(qry_ims)
        pos_features = self.forward_features(pos_ims)
        neg_features = self.forward_features(neg_ims)

        # Get predictions
        qry_preds = self.forward_head(qry_features)
        pos_preds = self.forward_head(pos_features)

        # Pool features for contrastive loss
        qry_pooled, pos_pooled, neg_pooled = self.get_feature_maps(
            [qry_features, pos_features, neg_features]
        )

        # Compute losses
        contrastive_loss = self.compute_contrastive_loss(
            qry_pooled, pos_pooled, neg_pooled
        )
        classification_loss = self.compute_classification_loss(
            qry_preds, pos_preds, qry_labels
        )

        total_loss = contrastive_loss + classification_loss

        # Compute metrics
        f1 = self.f1_score(qry_preds, qry_labels)
        stats = self.stat_scores(qry_preds, qry_labels)

        # Log metrics
        self.log(f"{stage}_loss", total_loss, prog_bar=True)
        self.log(f"{stage}_contrastive_loss", contrastive_loss)
        self.log(f"{stage}_classification_loss", classification_loss)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        self.log(f"{stage}_accuracy", stats[2] / stats[4])  # TP / (TP + TN + FP + FN)

        return {"loss": total_loss, "predictions": qry_preds, "targets": qry_labels}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        result = self.shared_step(batch, "train")
        return result["loss"]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        result = self.shared_step(batch, "val")
        return result["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        result = self.shared_step(batch, "test")
        return result["loss"]

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Prediction step."""
        qry_ims = batch["qry_im"]
        return self.forward(qry_ims)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
