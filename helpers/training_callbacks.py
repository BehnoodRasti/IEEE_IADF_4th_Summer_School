"""
Clean, modular training callbacks for the hyperspectral SSL tutorial.

This module provides professional-grade callbacks that separate concerns
and are easy to understand, maintain, and extend.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning import Callback
from IPython.display import display, clear_output
from typing import Dict, List, Optional, Any
from lightly.models import utils as L


from utils import denormalize_tensor, create_mae_reconstruction_plot


class SegmentationTrainingMonitor(Callback):
    """Clean callback for monitoring segmentation training with live plots."""
    
    def __init__(
        self, 
        plot_every_n_epochs: int = 1,
        n_preview_samples: int = 2,
        show_sanity_check: bool = False,
        figsize: tuple = (12, 4)
    ):
        """
        Initialize the training monitor.
        
        Args:
            plot_every_n_epochs: How often to show plots
            n_preview_samples: Number of validation samples to preview
            show_sanity_check: Whether to compute additional IoU sanity check
            figsize: Figure size for plots
        """
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.n_preview_samples = n_preview_samples
        self.show_sanity_check = show_sanity_check
        self.figsize = figsize
        
        # Metric storage
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_miou': []
        }
        
        # Temporary storage for current epoch
        self._current_epoch_losses = []

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset loss accumulator for new epoch."""
        self._current_epoch_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect batch losses for epoch averaging."""
        loss_value = self._extract_loss_value(outputs, trainer)
        if loss_value is not None:
            self._current_epoch_losses.append(loss_value)

    def on_train_epoch_end(self, trainer, pl_module):
        """Store average training loss for this epoch."""
        if self._current_epoch_losses:
            epoch_loss = np.mean(self._current_epoch_losses)
            # Store will happen in validation_epoch_end to keep everything synchronized
            self._last_train_loss = epoch_loss

    def on_validation_epoch_end(self, trainer, pl_module):
        """Main visualization and metric tracking logic."""
        # Skip sanity check
        if getattr(trainer, "sanity_checking", False):
            return
            
        epoch = trainer.current_epoch
        if epoch % self.plot_every_n_epochs != 0:
            return

        # Collect metrics
        self._update_metrics_history(trainer, epoch)
        
        # Clear output and create visualization
        clear_output(wait=True)
        self._create_training_dashboard(trainer, pl_module, epoch)

    def _extract_loss_value(self, outputs, trainer) -> Optional[float]:
        """Safely extract loss value from various output formats."""
        # Try direct tensor output
        if isinstance(outputs, torch.Tensor):
            return float(outputs.detach().cpu().item())
        
        # Try dictionary output    
        if isinstance(outputs, dict) and "loss" in outputs:
            return float(outputs["loss"].detach().cpu().item())
        
        # Try logged metrics as fallback
        metrics = trainer.callback_metrics
        for key in ["train_loss", "loss"]:
            if key in metrics:
                try:
                    value = metrics[key]
                    return float(value.detach().cpu().item() if torch.is_tensor(value) else value)
                except Exception:
                    continue
        
        return None

    def _update_metrics_history(self, trainer, epoch: int):
        """Update stored metrics with current epoch values."""
        # Get logged metrics
        logged_metrics = {k: float(v) for k, v in trainer.callback_metrics.items() 
                         if isinstance(v, (int, float, torch.Tensor))}
        
        # Store metrics
        self.metrics_history['epoch'].append(epoch)
        
        # Training loss
        train_loss = getattr(self, '_last_train_loss', None)
        self.metrics_history['train_loss'].append(train_loss)
        
        # Validation loss
        val_loss = logged_metrics.get('val_loss', None)
        self.metrics_history['val_loss'].append(val_loss)
        
        # Validation mIoU (find the right key)
        val_miou = self._find_miou_metric(logged_metrics)
        self.metrics_history['val_miou'].append(val_miou)

    def _find_miou_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        """Find the mIoU metric from logged metrics."""
        for key in metrics.keys():
            key_lower = key.lower()
            if key_lower.startswith('val_') and any(term in key_lower for term in ['jaccard', 'miou', 'iou']):
                return metrics[key]
        return None

    def _create_training_dashboard(self, trainer, pl_module, epoch: int):
        """Create the main training dashboard with metrics and samples."""
        print(f"📊 Epoch {epoch} — Training Progress Dashboard")
        
        # Create figure with just metrics (cleaner layout)
        fig = plt.figure(figsize=(12, 4))
        
        # Metrics plots
        self._plot_training_metrics(fig)
        
        plt.tight_layout()
        display(fig)
        plt.close(fig)
        
        # Display validation samples separately for better visibility
        self._plot_validation_samples_standalone(trainer, pl_module)
        
        # Optional sanity check
        if self.show_sanity_check:
            self._compute_sanity_check(trainer, pl_module)

    def _plot_training_metrics(self, fig):
        """Plot training and validation metrics."""
        epochs = self.metrics_history['epoch']
        if len(epochs) == 0:
            return
            
        # Loss plot
        ax1 = fig.add_subplot(1, 2, 1)
        
        train_losses = [x for x in self.metrics_history['train_loss'] if x is not None]
        val_losses = [x for x in self.metrics_history['val_loss'] if x is not None]
        
        if train_losses:
            train_epochs = epochs[:len(train_losses)]
            ax1.plot(train_epochs, train_losses, 'o-', label='Train Loss', color='blue')
        
        if val_losses:
            val_epochs = epochs[:len(val_losses)]
            ax1.plot(val_epochs, val_losses, 'o-', label='Val Loss', color='orange')
            
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # mIoU plot
        ax2 = fig.add_subplot(1, 2, 2)
        
        miou_values = [x for x in self.metrics_history['val_miou'] if x is not None]
        if miou_values:
            miou_epochs = epochs[:len(miou_values)]
            ax2.plot(miou_epochs, miou_values, 'o-', label='Val mIoU', color='green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mIoU')
            ax2.set_title('Validation mIoU')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No mIoU data yet', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Validation mIoU')

    def _plot_validation_samples_standalone(self, trainer, pl_module):
        """Display validation samples with predictions in standalone plots."""
        try:
            # Get validation batch
            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))
            
            # Get device and model predictions
            device = trainer.strategy.root_device
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            with torch.no_grad():
                logits = pl_module(images)
                predictions = logits.argmax(dim=1)
            
            # Get ignore index
            ignore_index = self._get_ignore_index(trainer, pl_module)
            
            # Show the requested number of samples
            n_samples = min(self.n_preview_samples, len(images))
            print(f"🔍 Validation Samples (showing {n_samples}/{len(images)}):")
            
            for i in range(n_samples):
                sample_dict = {
                    "image": images[i].detach().cpu(),
                    "mask": masks[i].detach().cpu(),
                }
                
                # Add prediction, masking ignore regions
                pred = predictions[i].detach().cpu()
                gt = masks[i].squeeze().cpu().long()
                
                if ignore_index is not None:
                    pred_masked = pred.clone()
                    pred_masked[gt == ignore_index] = ignore_index
                    sample_dict["prediction"] = pred_masked.unsqueeze(0)
                else:
                    sample_dict["prediction"] = pred.unsqueeze(0)
                
                # Use the datamodule's plotting function
                sample_fig = trainer.datamodule.plot(sample_dict)
                display(sample_fig)
                plt.close(sample_fig)
            
        except Exception as e:
            print(f"⚠️  Sample preview failed: {e}")

    def _get_ignore_index(self, trainer, pl_module) -> Optional[int]:
        """Get the ignore index for masking."""
        try:
            return trainer.datamodule.val_dataset.ignore_index
        except Exception:
            return getattr(pl_module, "ignore_index", None)

    def _compute_sanity_check(self, trainer, pl_module):
        """Compute additional IoU sanity check on fresh batch."""
        try:
            val_dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(val_dataloader))
            
            device = trainer.strategy.root_device
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            with torch.no_grad():
                logits = pl_module(images)
                predictions = logits.argmax(dim=1)
            
            ignore_index = self._get_ignore_index(trainer, pl_module)
            sanity_miou = compute_micro_iou(predictions.cpu(), masks.cpu(), ignore_index)
            
            print(f"🔍 Sanity Check - Micro IoU on fresh batch: {sanity_miou:.4f}")
            
        except Exception as e:
            print(f"⚠️  Sanity check failed: {e}")


def compute_micro_iou(predictions: torch.Tensor, targets: torch.Tensor, 
                     ignore_index: Optional[int] = None) -> float:
    """
    Compute micro-averaged IoU across all classes.
    
    Args:
        predictions: (B, H, W) predicted class indices
        targets: (B, H, W) or (B, 1, H, W) ground truth class indices  
        ignore_index: Class index to ignore in computation
        
    Returns:
        Micro-averaged IoU score
    """
    # Handle different target shapes
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    
    # Flatten to 1D
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Apply ignore mask
    if ignore_index is not None:
        valid_mask = targets_flat != ignore_index
        preds_flat = preds_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]
    
    if preds_flat.numel() == 0:
        return float('nan')
    
    # Get unique classes present in either predictions or targets
    all_classes = torch.unique(torch.cat([preds_flat.unique(), targets_flat.unique()]))
    
    total_intersection = 0
    total_union = 0
    
    for class_idx in all_classes:
        pred_mask = (preds_flat == class_idx)
        target_mask = (targets_flat == class_idx)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        total_intersection += intersection
        total_union += union
    
    return total_intersection / total_union if total_union > 0 else float('nan')

class MAEVisAndLossCallback(Callback):
    """Callback for MAE training visualization with live loss tracking."""

    def __init__(
        self,
        every_n_steps_images: int = 500,
        loss_every_n_steps: int = 50,
        rgb_indices=(60, 30, 10),
    ) -> None:
        super().__init__()
        self.every_n_steps_images = every_n_steps_images
        self.loss_every_n_steps = loss_every_n_steps
        self.rgb_indices = rgb_indices
        self.steps = []
        self.losses = []
        self._imgs = None

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step

        # Record loss
        loss_val = self._extract_loss(outputs, trainer)
        if loss_val is not None:
            self.steps.append(step)
            self.losses.append(loss_val)

        # Update images occasionally
        update_imgs = (self._imgs is None) or (step % self.every_n_steps_images == 0)
        if update_imgs:
            self._imgs = self._generate_reconstruction_images(pl_module, batch)

        # Draw visualization
        should_draw = update_imgs or (step % self.loss_every_n_steps == 0)
        if should_draw:
            self._draw_visualization(pl_module, step)

    def _extract_loss(self, outputs, trainer):
        """Extract loss value from various possible sources."""
        if isinstance(outputs, torch.Tensor):
            return float(outputs.detach().cpu().item())
        elif isinstance(outputs, dict) and "loss" in outputs:
            return float(outputs["loss"].detach().cpu().item())

        # Try from logged metrics
        for key in ("train_loss", "loss", "train_loss_step", "train/loss"):
            if key in trainer.callback_metrics:
                v = trainer.callback_metrics[key]
                try:
                    return float(v.detach().cpu().item() if torch.is_tensor(v) else float(v))
                except Exception:
                    continue
        return None

    def _generate_reconstruction_images(self, pl_module, batch):
        """Generate reconstruction images for visualization."""
        device = pl_module.device
        mean = pl_module.mean.to(device).float()
        std = pl_module.std.to(device).float()
        ps = pl_module.patch_size

        # Prepare input
        img = batch["image"][0:1].to(device).float()  # (1,C,H,W)
        B, C, H, W = img.shape
        img_norm = (img - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

        # Generate mask and reconstruct
        L_seq = pl_module.sequence_length
        idx_keep, idx_mask = L.random_token_mask(
            size=(B, L_seq), mask_ratio=pl_module.mask_ratio, device=device
        )

        x_encoded = pl_module.forward_encoder(images=img_norm, idx_keep=idx_keep)
        x_pred = pl_module.forward_decoder(x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)

        # Reconstruct full image
        patches = L.patchify(img_norm, ps)
        if pl_module.norm_pix:
            target_masked = L.get_at_index(patches, idx_mask - 1)
            mean_patch = target_masked.mean(dim=-1, keepdim=True)
            var_patch = target_masked.var(dim=-1, keepdim=True)
            x_pred_denorm = x_pred * (var_patch + 1e-6).sqrt() + mean_patch
        else:
            x_pred_denorm = x_pred

        recon_patches = L.set_at_index(patches, idx_mask - 1, x_pred_denorm)
        recon_norm = L.unpatchify(recon_patches, ps, channels=C)
        recon = denormalize_tensor(recon_norm, mean, std).squeeze(0)
        original = img.squeeze(0)

        # Create masked version for display
        masked = original.clone()
        Wp = W // ps
        for m in (idx_mask[0] - 1).detach().cpu().long().tolist():
            r, c = int(m // Wp), int(m % Wp)
            r0, r1 = r * ps, (r + 1) * ps
            c0, c1 = c * ps, (c + 1) * ps
            masked[:, r0:r1, c0:c1] = 0.0

        return original, masked, recon

    def _draw_visualization(self, pl_module, step):
        """Draw the complete visualization with images and loss."""
        clear_output(wait=True)
        fig = plt.figure(figsize=(14, 4.5), dpi=120)

        # Images
        if self._imgs is not None:
            original, masked, recon = self._imgs
            recon_fig = create_mae_reconstruction_plot(
                original, masked, recon, self.rgb_indices,
                pl_module.mask_ratio, step, figsize=(9, 4)
            )

            # Copy subplots to main figure
            recon_axes = recon_fig.get_axes()
            for i, recon_ax in enumerate(recon_axes):
                ax = fig.add_subplot(1, 4, i+1)
                img = recon_ax.get_images()[0].get_array()
                ax.imshow(img)
                ax.set_title(recon_ax.get_title())
                ax.axis("off")
            plt.close(recon_fig)

        # Loss plot
        ax4 = fig.add_subplot(1, 4, 4)
        if len(self.steps) > 0:
            ax4.plot(self.steps, self.losses)
            ax4.set_title("Training Loss")
            ax4.set_xlabel("Step")
            ax4.set_ylabel("MSE")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No loss data yet", ha="center", va="center")
            ax4.axis("off")

        fig.suptitle("MAE Training Progress", y=1.02)
        plt.tight_layout()
        display(fig)
        plt.close(fig)