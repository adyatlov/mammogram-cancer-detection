"""
EfficientNet-based model for breast cancer classification.
"""

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class MammogramClassifier(nn.Module):
    """
    EfficientNet-based classifier for mammogram images.

    Args:
        model_name: Name of the timm model to use
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (1 for binary)
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 1,
    ):
        super().__init__()

        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.backbone(x)


class MammogramLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training mammogram classifier.

    Args:
        model_name: Name of the timm model to use
        pretrained: Whether to use pretrained weights
        learning_rate: Initial learning rate
        pos_weight: Weight for positive class in BCE loss
        num_epochs: Total number of training epochs (for scheduler)
        label_smoothing: Smoothing for positive labels only (0.2 means 1.0 -> 0.8)
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        pos_weight: float = 1.0,
        num_epochs: int = 10,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MammogramClassifier(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=1,
        )

        # Weighted BCE loss for class imbalance
        self.pos_weight = torch.tensor([pos_weight])

        # Label smoothing for positive class only
        self.label_smoothing = label_smoothing

        # For tracking predictions during validation
        self.val_preds = []
        self.val_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss with optional positive label smoothing."""
        pos_weight = self.pos_weight.to(logits.device)

        # Apply label smoothing to positive labels only
        # e.g., label_smoothing=0.2 means: 1.0 -> 0.8, 0.0 -> 0.0
        if self.label_smoothing > 0:
            smoothed_labels = labels * (1.0 - self.label_smoothing)
        else:
            smoothed_labels = labels

        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            smoothed_labels,
            pos_weight=pos_weight,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss = self._compute_loss(logits, labels)

        # Log metrics
        preds = torch.sigmoid(logits.squeeze(-1))
        acc = ((preds > 0.5) == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        images = batch["image"]
        labels = batch["label"]

        logits = self(images)
        loss = self._compute_loss(logits, labels)

        preds = torch.sigmoid(logits.squeeze(-1))

        # Store for epoch-end metrics
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics at epoch end."""
        if not self.val_preds:
            return

        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)

        # Compute metrics
        acc = ((preds > 0.5) == labels).float().mean()

        # Compute pF1 (probabilistic F1)
        pf1 = self._compute_pf1(preds, labels)

        self.log("val_acc", acc)
        self.log("val_pf1", pf1, prog_bar=True)

        # Clear stored predictions
        self.val_preds = []
        self.val_labels = []

    def _compute_pf1(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute probabilistic F1 score.

        This is the competition metric - F1 score that works with probabilities.
        """
        # True positives, false positives, false negatives (soft versions)
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()

        # Precision and recall
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        # F1
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return f1

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.num_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
