"""
Utility functions for the Hyperspectral SSL Tutorial

This module contains common functions used throughout the tutorial notebook
to improve code readability and reduce repetition.
"""

import os
import random
import math
from typing import Tuple, List, Optional, Union
from functools import partial

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import rasterio

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

ENMAP_BANDS_CSV = "data/enmap_bands_remapped.csv"

def load_enmap_wavelengths(csv_path: str = ENMAP_BANDS_CSV) -> np.ndarray:
    """Load EnMAP wavelengths from CSV file.
    
    Args:
        csv_path: Path to CSV file containing band information
        
    Returns:
        Array of wavelengths in nanometers
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    order = np.argsort(data["idx"]) if "idx" in data.dtype.names else np.arange(len(data))
    wavelengths = np.asarray(data["center_nm"], dtype=float)[order]
    return wavelengths

# =============================================================================
# Visualization and Plotting
# =============================================================================

def clip_and_scale(arr: np.ndarray, low: float = 2, high: float = 98, 
                   axis: Optional[Union[int, Tuple[int, ...]]] = None, 
                   eps: float = 1e-8) -> np.ndarray:
    """
    Percentile-clip and scale array to [0,1] range.
    
    Args:
        arr: Input array
        low: Lower percentile for clipping
        high: Upper percentile for clipping  
        axis: Axes over which to compute percentiles
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Clipped and scaled array
    """
    lo = np.nanpercentile(arr, low, axis=axis, keepdims=True)
    hi = np.nanpercentile(arr, high, axis=axis, keepdims=True)
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + eps)

def insert_gaps_for_discontinuities(x_nm: np.ndarray, y: np.ndarray, 
                                   gap_factor: float = 2.0) -> np.ndarray:
    """
    Insert NaN values in spectral plots at wavelength discontinuities.
    
    Args:
        x_nm: Wavelength array
        y: Spectral values
        gap_factor: Factor to detect gaps (gaps > gap_factor * median_spacing)
        
    Returns:
        Modified spectral array with NaN at discontinuities
    """
    y_plot = y.copy()
    dx = np.diff(x_nm)
    if len(dx) == 0:
        return y_plot
    med = np.median(dx)
    if med <= 0:
        return y_plot
    large_gap_idx = np.where(dx > gap_factor * med)[0]
    for i in large_gap_idx:
        if i + 1 < len(y_plot):
            y_plot[i + 1] = np.nan
    return y_plot

def create_rgb_composite(image: torch.Tensor, rgb_indices: Tuple[int, int, int], 
                        contrast_percentiles: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """
    Create RGB composite from hyperspectral image.
    
    Args:
        image: Hyperspectral image tensor (C, H, W)
        rgb_indices: Indices for R, G, B bands
        contrast_percentiles: Percentiles for contrast stretching
        
    Returns:
        RGB image array (H, W, 3) in range [0, 1]
    """
    rgb = image[list(rgb_indices)].cpu().numpy().transpose(1, 2, 0)
    return clip_and_scale(rgb, low=contrast_percentiles[0], high=contrast_percentiles[1], 
                         axis=(0, 1))

def create_pca_composite(image: torch.Tensor, n_components: int = 3, 
                        contrast_percentiles: Tuple[float, float] = (2, 98)) -> np.ndarray:
    """
    Create PCA false-color composite from hyperspectral image.
    
    Args:
        image: Hyperspectral image tensor (C, H, W) 
        n_components: Number of PCA components to use
        contrast_percentiles: Percentiles for contrast stretching
        
    Returns:
        PCA composite array (H, W, 3) in range [0, 1]
    """
    C, H, W = image.shape
    X = image.cpu().numpy().reshape(C, -1).T  # [H*W, C]
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X).reshape(H, W, n_components)
    return clip_and_scale(pcs, low=contrast_percentiles[0], high=contrast_percentiles[1], 
                         axis=(0, 1))

def plot_spectral_windows(image: torch.Tensor, wavelengths: np.ndarray, 
                         windows: List[Tuple[str, float, float]], 
                         figsize_per_window: float = 3.2) -> plt.Figure:
    """
    Plot image data for specific spectral windows.
    
    Args:
        image: Hyperspectral image tensor (C, H, W)
        wavelengths: Wavelength array matching image channels
        windows: List of (name, min_wavelength, max_wavelength) tuples
        figsize_per_window: Figure width per window
        
    Returns:
        Matplotlib figure
    """
    img_np = image.cpu().numpy()  # [C,H,W]
    
    fig, axes = plt.subplots(1, len(windows), figsize=(figsize_per_window * len(windows), figsize_per_window))
    if len(windows) == 1:
        axes = [axes]
    
    for ax, (name, lo_nm, hi_nm) in zip(axes, windows):
        mask = (wavelengths >= lo_nm) & (wavelengths <= hi_nm)
        if not np.any(mask):
            ax.axis("off")
            ax.set_title(f"{name}\n(no bands)")
            continue
        band_slice = img_np[mask].mean(axis=0)  # average across window -> [H,W]
        band_disp = clip_and_scale(band_slice, low=2, high=98)  # robust contrast
        ax.imshow(band_disp, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    
    plt.tight_layout()
    return fig

def plot_hyperspectral_overview(image: torch.Tensor, wavelengths: np.ndarray, 
                               rgb_indices: Tuple[int, int, int],
                               pixel_locations: List[Tuple[int, int]] = None,
                               colors: List[str] = None,
                               figsize: Tuple[float, float] = (12, 4)) -> plt.Figure:
    """
    Create comprehensive overview plot: RGB, PCA false-color, and spectral signatures.
    
    Args:
        image: Hyperspectral image tensor (C, H, W)
        wavelengths: Wavelength array matching image channels  
        rgb_indices: Indices for R, G, B bands
        pixel_locations: List of (y, x) coordinates for spectral signatures
        colors: Colors for spectral signature lines
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    H, W = image.shape[1], image.shape[2]
    
    if pixel_locations is None:
        pixel_locations = [(32, 32), (32, W-32), (H-32, 32), (H-32, W-32)]
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1) RGB composite
    rgb = create_rgb_composite(image, rgb_indices)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Composite")
    axes[0].axis("off")
    
    # 2) PCA false-color  
    pca_rgb = create_pca_composite(image)
    axes[1].imshow(pca_rgb)
    axes[1].set_title("PCA False-Color")
    axes[1].axis("off")
    
    # 3) Spectral signatures
    for (y, x), c in zip(pixel_locations, colors):
        if 0 <= y < H and 0 <= x < W:
            spectrum = image[:, y, x].cpu().numpy()
            y_plot = insert_gaps_for_discontinuities(wavelengths, spectrum)
            axes[2].plot(wavelengths, y_plot, color=c, label=f'Pixel ({y},{x})', linewidth=2)
    
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].set_ylabel("Reflectance")
    axes[2].set_title("Spectral Signatures")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# Temporal Data Visualization  
# =============================================================================

def extract_tile_id_from_filename(path: str) -> str:
    """
    Extract tile ID from filename (basename without extension).
    
    Args:
        path: File path (e.g., '/path/to/patch_dir/{tile_id}.tif')
        
    Returns:
        Tile ID string (e.g., 'tile_123' from 'tile_123.tif')
    """
    return os.path.splitext(os.path.basename(path))[0]

def plot_temporal_rgb_gallery(images: List[torch.Tensor], labels: List[str], 
                             rgb_indices: Tuple[int, int, int],
                             max_cols: int = 5,
                             figsize_per_image: float = 3.3,
                             patch_id: str = None) -> plt.Figure:
    """
    Plot RGB gallery of multiple timestamps.
    
    Args:
        images: List of hyperspectral image tensors
        labels: List of labels for each timestamp
        rgb_indices: Indices for R, G, B bands
        max_cols: Maximum columns in grid
        figsize_per_image: Size per image subplot
        patch_id: Optional patch identifier for title
        
    Returns:
        Matplotlib figure
    """
    n = len(images)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_image * ncols, figsize_per_image * nrows))
    axes = np.atleast_1d(axes).ravel()
    
    # Compute global RGB percentiles for consistent coloring
    all_rgb = np.stack([create_rgb_composite(img, rgb_indices, (0, 100)) for img in images], axis=0)
    global_lo = np.nanpercentile(all_rgb, 2, axis=(0, 1, 2))
    global_hi = np.nanpercentile(all_rgb, 98, axis=(0, 1, 2))
    
    def scale_rgb_global(rgb_arr):
        return np.clip((rgb_arr - global_lo) / (global_hi - global_lo + 1e-8), 0, 1)
    
    for ax, img, lab in zip(axes, images, labels):
        rgb = img[list(rgb_indices)].numpy().transpose(1, 2, 0)
        rgb_disp = scale_rgb_global(rgb)
        ax.imshow(rgb_disp)
        ax.set_title(lab, fontsize=9)
        ax.axis("off")
    
    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")
    
    title = f"Temporal RGB Gallery"
    if patch_id:
        title += f" — Patch {patch_id}"
    if n > 1:
        title += f" ({n} timestamps)"
    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    return fig

def plot_temporal_spectra(images: List[torch.Tensor], labels: List[str], 
                         wavelengths: np.ndarray, pixel_yx: Tuple[int, int],
                         patch_id: str = None,
                         figsize: Tuple[float, float] = (7, 4)) -> plt.Figure:
    """
    Plot spectral signatures over time for a specific pixel.
    
    Args:
        images: List of hyperspectral image tensors
        labels: List of labels for each timestamp  
        wavelengths: Wavelength array
        pixel_yx: (y, x) coordinates of pixel
        patch_id: Optional patch identifier for title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    py, px = pixel_yx
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for img, lab in zip(images, labels):
        spec = img[:, py, px].numpy()
        y_plot = insert_gaps_for_discontinuities(wavelengths, spec)
        ax.plot(wavelengths, y_plot, label=lab, linewidth=2)
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    
    title = f"Temporal Spectra at pixel (y={py}, x={px})"
    if patch_id:
        title += f" — Patch {patch_id}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2, fontsize=8)
    plt.tight_layout()
    return fig

# =============================================================================
# Dataset Visualization
# =============================================================================

def create_class_colormap_and_legend(allowed_classes: set, class_mapping: dict) -> Tuple[list, dict]:
    """
    Create colormap and legend for classification datasets.
    
    Args:
        allowed_classes: Set of allowed class codes
        class_mapping: Dictionary mapping class codes to names
        
    Returns:
        Tuple of (legend_list, color_map_dict)
    """
    non_bg = sorted([c for c in allowed_classes if c != 0])
    cmap = cm.get_cmap("tab20", max(1, len(non_bg)))
    color_map = {0: (0.0, 0.0, 0.0)}  # background = black
    
    for i, cls in enumerate(non_bg):
        r, g, b, _ = cmap(i % max(1, len(non_bg)))
        color_map[cls] = (r, g, b)
    
    legend = []
    for cls in sorted(allowed_classes):
        name = class_mapping.get(cls, f"Class {cls}") if cls != 0 else "Background"
        legend.append((cls, name, color_map[cls]))
    
    return legend, color_map

def plot_classification_sample(sample: dict, rgb_indices: Tuple[int, int, int], 
                              color_map: dict, allowed_classes: set,
                              title_prefix: str = "Sample") -> plt.Figure:
    """
    Plot RGB image and colorized classification mask.
    
    Args:
        sample: Dataset sample with 'image' and 'mask' keys
        rgb_indices: Indices for R, G, B bands
        color_map: Mapping from class codes to RGB colors
        allowed_classes: Set of allowed class codes
        title_prefix: Prefix for subplot titles
        
    Returns:
        Matplotlib figure
    """
    # Create RGB composite
    rgb_tensor = sample["image"][list(rgb_indices)]
    if rgb_tensor.dim() == 3:  # (C, H, W)
        rgb = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    else:  # Already (H, W, C)
        rgb = rgb_tensor.cpu().numpy()
    
    vmin, vmax = float(np.nanmin(rgb)), float(np.nanmax(rgb))
    if vmax > vmin:
        rgb = (rgb - vmin) / (vmax - vmin)
    else:
        rgb = np.zeros_like(rgb)
    
    # Process mask
    raw_mask = sample["mask"].squeeze().cpu().numpy()
    mask = np.where(np.isin(raw_mask, list(allowed_classes)), raw_mask, 0)
    
    # Colorize mask
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    for cls in np.unique(mask):
        mask_rgb[mask == cls] = color_map.get(int(cls), (1.0, 0.0, 1.0))  # magenta if unexpected
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"{title_prefix} RGB")
    axes[0].axis("off")
    
    axes[1].imshow(mask_rgb)
    axes[1].set_title(f"{title_prefix} Mask") 
    axes[1].axis("off")
    
    return fig

# =============================================================================
# MAE Visualization Utilities
# =============================================================================

def denormalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Reverse channel-wise normalization for (B,C,H,W) tensor.
    
    Args:
        x: Normalized tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        De-normalized tensor
    """
    return x * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def create_mae_reconstruction_plot(original: torch.Tensor, masked: torch.Tensor, 
                                  reconstruction: torch.Tensor, rgb_indices: Tuple[int, int, int],
                                  mask_ratio: float, step: int,
                                  figsize: Tuple[float, float] = (12, 4)) -> plt.Figure:
    """
    Create MAE reconstruction visualization plot.
    
    Args:
        original: Original image tensor (C, H, W)
        masked: Masked image tensor (C, H, W)  
        reconstruction: Reconstructed image tensor (C, H, W)
        rgb_indices: Indices for R, G, B bands
        mask_ratio: Masking ratio used
        step: Training step number
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Compute shared RGB scaling from original
    bands_ref = torch.stack([
        original[rgb_indices[0]], original[rgb_indices[1]], original[rgb_indices[2]]
    ], dim=0).cpu().numpy()
    p2 = np.percentile(bands_ref, 2, axis=(1, 2))
    p98 = np.percentile(bands_ref, 98, axis=(1, 2))
    
    def _rgb_composite_shared(img_tensor, p2, p98):
        bands = torch.stack([
            img_tensor[rgb_indices[0]], img_tensor[rgb_indices[1]], img_tensor[rgb_indices[2]]
        ], dim=0).cpu().numpy()
        rgb = []
        for i in range(3):
            ch = np.clip(bands[i], p2[i], p98[i])
            denom = max(p98[i] - p2[i], 1e-6)
            ch = (ch - p2[i]) / denom
            rgb.append(ch)
        return np.stack(rgb, axis=-1).astype(np.float32)
    
    rgb_orig = _rgb_composite_shared(original, p2, p98)
    rgb_masked = _rgb_composite_shared(masked, p2, p98)
    rgb_recon = _rgb_composite_shared(reconstruction, p2, p98)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(rgb_orig)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(rgb_masked)
    axes[1].set_title(f"Masked ({mask_ratio:.0%})")
    axes[1].axis("off")
    
    axes[2].imshow(rgb_recon)
    axes[2].set_title(f"Reconstruction @ step {step}")
    axes[2].axis("off")
    
    fig.suptitle("MAE Reconstruction Progress")
    plt.tight_layout()
    return fig

# =============================================================================
# PCA Analysis Utilities (for Exercise 2)
# =============================================================================

def plot_pca_explained_variance(pca_model, n_components_to_show: int = 50, 
                               figsize: Tuple[float, float] = (12, 4)) -> plt.Figure:
    """
    Plot explained variance ratio and cumulative explained variance for PCA.
    
    Args:
        pca_model: Fitted sklearn PCA model
        n_components_to_show: Number of components to show in plots
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    explained_var = pca_model.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    n_show = min(n_components_to_show, len(explained_var))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual explained variance
    ax1.bar(range(1, n_show + 1), explained_var[:n_show], alpha=0.7, color='skyblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Component Variance')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    ax2.plot(range(1, n_show + 1), cumulative_var[:n_show], 'o-', color='orange', linewidth=2)
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.axhline(y=0.99, color='darkred', linestyle='--', alpha=0.7, label='99% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def plot_pca_eigenspectra(pca_model, wavelengths: np.ndarray, 
                         n_components_to_plot: int = 6,
                         figsize: Tuple[float, float] = (15, 8)) -> plt.Figure:
    """
    Plot the first few PCA components as 'eigenspectra'.
    
    Args:
        pca_model: Fitted sklearn PCA model
        wavelengths: Wavelength values for x-axis
        n_components_to_plot: Number of components to visualize
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    components = pca_model.components_
    n_plot = min(n_components_to_plot, components.shape[0])
    
    # Determine grid layout
    ncols = 3
    nrows = math.ceil(n_plot / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_plot))
    
    for i in range(n_plot):
        ax = axes[i]
        component = components[i]
        
        # Plot the component as a spectrum
        ax.plot(wavelengths, component, color=colors[i], linewidth=2)
        ax.set_title(f'PC{i+1} (Eigenspectrum)\n'
                    f'Var: {pca_model.explained_variance_ratio_[i]:.3f}')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Loading')
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)
    
    # Hide empty subplots
    for i in range(n_plot, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Principal Component Analysis: Eigenspectra', fontsize=16)
    plt.tight_layout()
    return fig

def analyze_pca_spectral_preservation(original_spectra: np.ndarray, 
                                    reconstructed_spectra: np.ndarray,
                                    wavelengths: np.ndarray,
                                    n_samples_to_plot: int = 3,
                                    figsize: Tuple[float, float] = (15, 5)) -> plt.Figure:
    """
    Analyze how well PCA preserves spectral signatures.
    
    Args:
        original_spectra: Original spectra (N, C)
        reconstructed_spectra: PCA reconstructed spectra (N, C) 
        wavelengths: Wavelength values
        n_samples_to_plot: Number of example spectra to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure with comparison plots
    """
    # Calculate reconstruction error metrics
    mse_per_sample = np.mean((original_spectra - reconstructed_spectra) ** 2, axis=1)
    mae_per_sample = np.mean(np.abs(original_spectra - reconstructed_spectra), axis=1)
    
    # Select samples for visualization (best, worst, median)
    sample_indices = [
        np.argmin(mse_per_sample),  # Best reconstruction
        np.argsort(mse_per_sample)[len(mse_per_sample)//2],  # Median
        np.argmax(mse_per_sample)   # Worst reconstruction
    ]
    sample_indices = sample_indices[:n_samples_to_plot]
    
    fig, axes = plt.subplots(1, n_samples_to_plot, figsize=figsize)
    if n_samples_to_plot == 1:
        axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Plot original and reconstructed spectra
        ax.plot(wavelengths, original_spectra[idx], 'b-', linewidth=2, 
               label=f'Original', alpha=0.8)
        ax.plot(wavelengths, reconstructed_spectra[idx], 'r--', linewidth=2, 
               label=f'PCA Reconstructed', alpha=0.8)
        
        # Add error information
        mse = mse_per_sample[idx]
        mae = mae_per_sample[idx]
        
        quality = ['Best', 'Median', 'Worst'][i] if n_samples_to_plot == 3 else f'Sample {i+1}'
        ax.set_title(f'{quality} Reconstruction\nMSE: {mse:.4f}, MAE: {mae:.4f}')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)
    
    plt.suptitle('PCA Spectral Preservation Analysis', fontsize=16)
    plt.tight_layout()
    return fig

def compare_training_curves(results_dict: dict, figsize: Tuple[float, float] = (15, 5)) -> plt.Figure:
    """
    Compare training curves across different PCA configurations.
    
    Args:
        results_dict: Dictionary with structure:
                     {config_name: {'epochs': [...], 'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}}
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for i, (config_name, metrics) in enumerate(results_dict.items()):
        color = colors[i]
        epochs = metrics['epochs']
        
        # Plot losses
        if 'train_loss' in metrics:
            ax1.plot(epochs, metrics['train_loss'], '-', color=color, 
                    label=f'{config_name} (Train)', linewidth=2, alpha=0.8)
        if 'val_loss' in metrics:
            ax1.plot(epochs, metrics['val_loss'], '--', color=color, 
                    label=f'{config_name} (Val)', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy/metrics
    for i, (config_name, metrics) in enumerate(results_dict.items()):
        color = colors[i]
        epochs = metrics['epochs']
        
        metric_key = 'val_acc' if 'val_acc' in metrics else 'val_miou'
        if metric_key in metrics:
            ax2.plot(epochs, metrics[metric_key], 'o-', color=color, 
                    label=config_name, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Metric')
    ax2.set_title('Validation Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# Self-Supervised Learning Analysis Utilities (for Exercise 4)
# =============================================================================

def monitor_representation_collapse(embeddings: torch.Tensor, 
                                   step: int,
                                   collapse_threshold: float = 0.01) -> dict:
    """
    Monitor for representation collapse in self-supervised learning.
    
    Args:
        embeddings: Tensor of shape (N, D) with N samples and D dimensions
        step: Current training step
        collapse_threshold: Threshold below which std indicates collapse
        
    Returns:
        Dictionary with collapse metrics
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Calculate various collapse indicators
    std_per_dim = np.std(embeddings_np, axis=0)
    mean_std = np.mean(std_per_dim)
    min_std = np.min(std_per_dim)
    
    # Calculate pairwise cosine similarities
    embeddings_norm = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    # Remove diagonal (self-similarity)
    mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool)
    similarities = similarity_matrix[mask]
    mean_similarity = torch.mean(similarities).item()
    
    # Effective rank (indicator of dimensionality usage)
    _, s, _ = torch.svd(embeddings)
    s_normalized = s / torch.sum(s)
    effective_rank = torch.exp(-torch.sum(s_normalized * torch.log(s_normalized + 1e-8))).item()
    
    metrics = {
        'step': step,
        'mean_std': mean_std,
        'min_std': min_std,
        'mean_similarity': mean_similarity,
        'effective_rank': effective_rank,
        'is_collapsed': mean_std < collapse_threshold
    }
    
    return metrics

def plot_collapse_monitoring(collapse_history: List[dict], 
                           figsize: Tuple[float, float] = (15, 4)) -> plt.Figure:
    """
    Plot collapse monitoring metrics over training.
    
    Args:
        collapse_history: List of collapse metric dictionaries
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not collapse_history:
        return None
    
    steps = [m['step'] for m in collapse_history]
    mean_stds = [m['mean_std'] for m in collapse_history]
    mean_sims = [m['mean_similarity'] for m in collapse_history]
    eff_ranks = [m['effective_rank'] for m in collapse_history]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Standard deviation plot
    ax1.plot(steps, mean_stds, 'b-', linewidth=2, label='Mean Std')
    ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Collapse Threshold')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Embedding Standard Deviation')
    ax1.set_title('Representation Diversity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Similarity plot
    ax2.plot(steps, mean_sims, 'g-', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Mean Cosine Similarity')
    ax2.set_title('Representation Similarity')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    # Effective rank plot
    ax3.plot(steps, eff_ranks, 'purple', linewidth=2)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Effective Rank')
    ax3.set_title('Dimensionality Usage')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def compare_ssl_embeddings(embeddings_dict: dict, labels: np.ndarray = None,
                          method: str = 'tsne', figsize: Tuple[float, float] = (15, 5)) -> plt.Figure:
    """
    Compare embeddings from different SSL methods using dimensionality reduction.
    
    Args:
        embeddings_dict: Dictionary {method_name: embeddings_tensor}
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('tsne' or 'pca')
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_methods = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    for i, (method_name, embeddings) in enumerate(embeddings_dict.items()):
        ax = axes[i]
        
        # Apply dimensionality reduction
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        
        embeddings_np = embeddings.detach().cpu().numpy()
        embeddings_2d = reducer.fit_transform(embeddings_np)
        
        # Plot
        if labels is not None:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7, s=20)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      alpha=0.7, s=20, color='blue')
        
        ax.set_title(f'{method_name} Embeddings')
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_spectral_ssl_learning(original_images: torch.Tensor,
                                 embeddings: torch.Tensor, 
                                 wavelengths: np.ndarray,
                                 n_samples: int = 5,
                                 figsize: Tuple[float, float] = (15, 8)) -> plt.Figure:
    """
    Analyze what spectral features SSL methods learn to emphasize.
    
    Args:
        original_images: Original hyperspectral images (N, C, H, W)
        embeddings: Learned embeddings (N, D)
        wavelengths: Wavelength values for spectral bands
        n_samples: Number of samples to analyze
        figsize: Figure size
        
    Returns:
        Matplotlib figure showing spectral analysis
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Get a subset of samples
    n_samples = min(n_samples, original_images.size(0))
    indices = np.random.choice(original_images.size(0), n_samples, replace=False)
    
    images_subset = original_images[indices]  # (n_samples, C, H, W)
    embeddings_subset = embeddings[indices]  # (n_samples, D)
    
    # Calculate average spectrum per sample
    avg_spectra = torch.mean(images_subset, dim=(2, 3))  # (n_samples, C)
    avg_spectra_np = avg_spectra.detach().cpu().numpy()
    
    # PCA on embeddings to find main directions
    embeddings_np = embeddings_subset.detach().cpu().numpy()
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_np)
    pca_emb = PCA(n_components=2)
    embeddings_pca = pca_emb.fit_transform(embeddings_scaled)
    
    # PCA on spectra
    pca_spec = PCA(n_components=2)
    spectra_pca = pca_spec.fit_transform(avg_spectra_np)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Embeddings PCA
    axes[0, 0].scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                      c=range(n_samples), cmap='viridis', s=100)
    axes[0, 0].set_title('SSL Embeddings (PCA)')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spectral PCA
    axes[0, 1].scatter(spectra_pca[:, 0], spectra_pca[:, 1], 
                      c=range(n_samples), cmap='viridis', s=100)
    axes[0, 1].set_title('Average Spectra (PCA)')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Spectral signatures
    for i in range(n_samples):
        axes[1, 0].plot(wavelengths, avg_spectra_np[i], 
                       label=f'Sample {i+1}', alpha=0.8, linewidth=2)
    axes[1, 0].set_xlabel('Wavelength (nm)')
    axes[1, 0].set_ylabel('Reflectance')
    axes[1, 0].set_title('Average Spectral Signatures')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Embedding vs spectral correlation
    # Calculate correlation between embedding distances and spectral distances
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr
    
    emb_distances = pdist(embeddings_np)
    spec_distances = pdist(avg_spectra_np)
    
    correlation, p_value = pearsonr(emb_distances, spec_distances)
    
    axes[1, 1].scatter(spec_distances, emb_distances, alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Spectral Distance')
    axes[1, 1].set_ylabel('Embedding Distance')
    axes[1, 1].set_title(f'Spectral vs Embedding Distance\\nCorr: {correlation:.3f} (p={p_value:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# Embedding Visualization Utilities (for Section 4)
# =============================================================================

def extract_embeddings_from_model(model, dataloader, device='cpu', max_samples=None, 
                                  normalization_stats=None):
    """
    Extract embeddings from a pre-trained model.
    
    Args:
        model: Pre-trained model (should have a backbone attribute)
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        max_samples: Maximum number of samples to process (None for all)
        normalization_stats: Tuple of (mean, std) numpy arrays for normalization
        
    Returns:
        Tuple of (embeddings, sample_images) where embeddings is numpy array
    """
    model.eval()
    model = model.to(device)
    
    embeddings = []
    sample_images = []  # Will store individual tensors
    total_processed = 0
    
    # Convert normalization stats to tensors if provided
    if normalization_stats is not None:
        mean_np, std_np = normalization_stats
        mean_tensor = torch.from_numpy(mean_np).float().to(device)
        std_tensor = torch.from_numpy(std_np).float().to(device)
        print(f"  Using normalization - Mean: {mean_np[:3]}, Std: {std_np[:3]}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and total_processed >= max_samples:
                break
                
            images = batch["image"].to(device)
            
            # CRITICAL: Apply preprocessing (float conversion + normalization)
            if normalization_stats is not None:
                # Convert to float and normalize
                images = images.float()
                images = (images - mean_tensor.view(1, -1, 1, 1)) / std_tensor.view(1, -1, 1, 1)
            else:
                # At minimum, convert to float
                images = images.float()
            
            # Get embeddings from the backbone
            if hasattr(model, 'backbone'):
                backbone_features = model.backbone(images)
            else:
                backbone_features = model(images)
            
            # Handle different output formats
            if len(backbone_features.shape) == 3:  # [B, seq_len, dim] (ViT)
                features = backbone_features[:, 0, :]  # Take [CLS] token
            elif len(backbone_features.shape) == 4:  # [B, C, H, W] (CNN)
                features = torch.mean(backbone_features, dim=(2, 3))  # Global average pooling
            else:  # [B, dim]
                features = backbone_features
            
            embeddings.append(features.cpu())
            
            # Store sample images for cluster analysis (store all for better representation)
            batch_original = batch["image"].float()
            sample_images.extend([img for img in batch_original])
            
            # Limit to max_samples if specified
            if max_samples and len(sample_images) > max_samples:
                sample_images = sample_images[:max_samples]
            
            total_processed += len(images)
            
            if batch_idx % 10 == 0:
                print(f"  Processed {total_processed} samples...")
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    
    # Convert sample_images list to tensor if we have images
    if sample_images:
        sample_images = torch.stack(sample_images)
    
    return embeddings, sample_images

def create_tsne_visualization(embeddings, sample_images=None, rgb_indices=(60, 30, 10),
                             method_name="SSL Model", perplexity=30, 
                             figsize=(16, 12), max_pca_components=50):
    """
    Create comprehensive t-SNE visualization of embeddings.
    
    Args:
        embeddings: Numpy array of embeddings (N, D)
        sample_images: Optional sample images for display
        rgb_indices: RGB band indices for visualization
        method_name: Name of the SSL method for titles
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
        max_pca_components: Max PCA components before t-SNE
        
    Returns:
        Matplotlib figure and 2D embeddings
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    print(f"🔬 Creating t-SNE visualization for {len(embeddings)} samples...")
    
    # PCA preprocessing for efficiency
    n_pca = min(max_pca_components, embeddings.shape[1], len(embeddings) - 1)
    print(f"  Step 1: PCA to {n_pca} dimensions...")
    pca = PCA(n_components=n_pca, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # t-SNE
    print(f"  Step 2: t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
        verbose=0
    )
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # Create visualization
    fig = plt.figure(figsize=figsize)
    
    # Main t-SNE plot
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
    
    # Color by sample index with density-based coloring
    scatter = ax_main.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1],
        c=np.arange(len(embeddings_2d)),
        cmap='viridis',
        alpha=0.7,
        s=25,
        edgecolors='black',
        linewidths=0.1
    )
    
    ax_main.set_title(f"t-SNE Visualization: {method_name} Embeddings\\n" +
                     f"Hyperspectral Data ({len(embeddings_2d)} samples)", fontsize=14)
    ax_main.set_xlabel("t-SNE Dimension 1")
    ax_main.set_ylabel("t-SNE Dimension 2")
    ax_main.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label("Sample Index", rotation=270, labelpad=15)
    
    # Sample images if provided
    if sample_images is not None:
        n_samples_show = min(4, len(sample_images))
        positions = [
            (0, 3), (1, 3), (2, 0), (2, 1)
        ]
        
        for i in range(n_samples_show):
            ax_img = plt.subplot2grid((3, 4), positions[i], rowspan=1)
            
            # Create RGB composite
            if len(sample_images[i].shape) == 3:  # (C, H, W)
                rgb_image = sample_images[i][list(rgb_indices)].numpy().transpose(1, 2, 0)
            else:  # Already (H, W, C)
                rgb_image = sample_images[i].numpy()
            
            # Clip and normalize for display (using percentile scaling)
            p2, p98 = np.percentile(rgb_image, [2, 98])
            rgb_image = np.clip((rgb_image - p2) / (p98 - p2 + 1e-8), 0, 1)
            
            ax_img.imshow(rgb_image)
            ax_img.set_title(f"Sample {i+1}", fontsize=10)
            ax_img.axis("off")
    
    # Statistics panel
    ax_stats = plt.subplot2grid((3, 4), (2, 2), rowspan=1, colspan=2)
    ax_stats.axis("off")
    
    stats_text = f"""
Embedding Analysis:
• Original dimensions: {embeddings.shape[1]}
• PCA dimensions: {embeddings_pca.shape[1]}
• PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}
• t-SNE perplexity: {perplexity}
• Samples visualized: {len(embeddings_2d)}

Interpretation Guide:
• Clusters → Similar spectral/spatial content
• Smooth transitions → Meaningful gradients  
• Isolated points → Unique/rare samples
• Overall spread → Model's discriminative power
"""
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig, embeddings_2d

def analyze_embedding_clusters(embeddings_2d, n_clusters=5, method='kmeans'):
    """
    Analyze clusters in 2D embeddings and provide insights.
    
    Args:
        embeddings_2d: 2D t-SNE embeddings
        n_clusters: Number of clusters to identify
        method: Clustering method ('kmeans' or 'dbscan')
        
    Returns:
        Cluster labels and analysis results
    """
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(embeddings_2d)
    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clusterer.fit_predict(embeddings_2d)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    # Calculate cluster statistics
    unique_labels = np.unique(cluster_labels)
    cluster_stats = {}
    
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_points = embeddings_2d[mask]
        cluster_stats[label] = {
            'size': np.sum(mask),
            'center': np.mean(cluster_points, axis=0),
            'std': np.std(cluster_points, axis=0),
            'percentage': np.sum(mask) / len(embeddings_2d) * 100
        }
    
    print(f"🔍 Identified {n_clusters} clusters:")
    for label, stats in cluster_stats.items():
        if label == -1:
            print(f"  Noise: {stats['size']} samples ({stats['percentage']:.1f}%)")
        else:
            print(f"  Cluster {label}: {stats['size']} samples ({stats['percentage']:.1f}%)")
    
    return cluster_labels, cluster_stats

# =============================================================================
# Interactive Plotting Utilities
# =============================================================================

def create_interactive_spectral_plot(image: torch.Tensor, wavelengths: np.ndarray,
                                    rgb_indices: Tuple[int, int, int],
                                    figsize: Tuple[float, float] = (16, 4)) -> plt.Figure:
    """
    Create interactive spectral plot (requires %matplotlib widget).
    
    Args:
        image: Hyperspectral image tensor (C, H, W)
        wavelengths: Wavelength array
        rgb_indices: Indices for R, G, B bands  
        figsize: Figure size
        
    Returns:
        Matplotlib figure with click handler attached
    """
    H, W = image.shape[1], image.shape[2]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # RGB composite
    rgb = create_rgb_composite(image, rgb_indices)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Composite (Click a pixel)")
    axes[0].axis("off")
    
    # PCA false-color
    pca_rgb = create_pca_composite(image)
    axes[1].imshow(pca_rgb)
    axes[1].set_title("PCA False-Color") 
    axes[1].axis("off")
    
    # Initial spectrum
    initial_y, initial_x = H // 2, W // 2
    spectrum = image[:, initial_y, initial_x].cpu().numpy()
    y_plot = insert_gaps_for_discontinuities(wavelengths, spectrum)
    
    line, = axes[2].plot(wavelengths, y_plot, color='royalblue')
    axes[2].set_xlabel("Wavelength (nm)")
    axes[2].set_ylabel("Reflectance")
    axes[2].set_title(f"Spectrum at ({initial_x}, {initial_y})")
    axes[2].grid(True, alpha=0.4)
    axes[2].set_ylim(image.cpu().numpy().min(), image.cpu().numpy().max())
    
    # Add marker
    marker, = axes[0].plot(initial_x, initial_y, 'r+', markersize=10, markeredgewidth=2)
    
    def on_click(event):
        if event.inaxes != axes[0]:
            return
        x, y = int(event.xdata), int(event.ydata)
        
        if 0 <= x < W and 0 <= y < H:
            marker.set_data([x], [y])
            new_spectrum = image[:, y, x].cpu().numpy()
            new_y_plot = insert_gaps_for_discontinuities(wavelengths, new_spectrum)
            line.set_ydata(new_y_plot)
            axes[2].set_title(f"Spectrum at ({x}, {y})")
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    return fig


# Create an interactive temporal plot utility function
def create_interactive_temporal_plot(images, tile_ids, wavelengths, rgb_indices, patch_id):
    """Create interactive temporal visualization with clickable first image."""
    from functools import partial
    import math
    from utils import clip_and_scale, insert_gaps_for_discontinuities

    n = len(images)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    C, H, W = images[0].shape

    # Compute shared RGB scaling for visual consistency
    all_rgb = np.stack([img[list(rgb_indices)].numpy().transpose(1, 2, 0) for img in images], axis=0)
    lo = np.nanpercentile(all_rgb, 2, axis=(0, 1, 2))
    hi = np.nanpercentile(all_rgb, 98, axis=(0, 1, 2))

    def scale_rgb(x_hw3):
        return np.clip((x_hw3 - lo) / (hi - lo + 1e-8), 0, 1)

    # Create figure layout
    fig = plt.figure(figsize=(3.3 * ncols, 3.3 * nrows + 3.8))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[*[2]*nrows, 2.3])

    # Create image subplots
    image_axes = []
    for i in range(nrows * ncols):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        if i < n:
            rgb = images[i][list(rgb_indices)].numpy().transpose(1, 2, 0)
            ax.imshow(scale_rgb(rgb))
            title = tile_ids[i] if i < len(tile_ids) else f"T{i+1}"
            ax.set_title(title, fontsize=9)
        ax.axis("off")
        image_axes.append(ax)

    # Create spectral plot
    ax_spec = fig.add_subplot(gs[nrows, :])
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Reflectance")
    ax_spec.grid(True, alpha=0.3)

    # Initialize with center pixel
    py, px = H // 2, W // 2
    lines = []
    for img, tile_id in zip(images, tile_ids):
        spec = img[:, py, px].numpy()
        y_plot = insert_gaps_for_discontinuities(wavelengths, spec)
        line, = ax_spec.plot(wavelengths, y_plot, label=tile_id, linewidth=2)
        lines.append(line)

    ax_spec.legend(loc="best", ncol=2, fontsize=8)
    ax_spec.set_title(f"Spectra over time at (y={py}, x={px}) — Patch {patch_id}")
    ax_spec.margins(y=0.05)

    # Add marker to first image
    marker, = image_axes[0].plot([px], [py], 'r+', markersize=10, markeredgewidth=2)

    # Click handler
    def on_click(event):
        if event.inaxes != image_axes[0] or event.xdata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if not (0 <= x < W and 0 <= y < H):
            return

        marker.set_data([x], [y])
        for line, img in zip(lines, images):
            spec = img[:, y, x].numpy()
            y_plot = insert_gaps_for_discontinuities(wavelengths, spec)
            line.set_ydata(y_plot)

        ax_spec.set_title(f"Spectra over time at (y={y}, x={x}) — Patch {patch_id}")
        ax_spec.relim()
        ax_spec.autoscale_view(scaley=True)
        ax_spec.margins(y=0.05)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    return fig

# =============================================================================
# Common Spectral Windows
# =============================================================================

SPECTRAL_WINDOWS = [
    ("Blue 450–490", 450, 490),
    ("Green 540–580", 540, 580), 
    ("Red 650–680", 650, 680),
    ("Red Edge 705–740", 705, 740),
    ("NIR 800–880", 800, 880),
    ("SWIR1 1600–1700", 1600, 1700),
]
