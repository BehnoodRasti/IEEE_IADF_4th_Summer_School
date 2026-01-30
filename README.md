# Self-Supervised Learning for Multi- and Hyperspectral Foundation Models
## A Comprehensive Tutorial on MAE Pre-training and Downstream Tasks

## Overview
This repository provides a hands-on tutorial on **self-supervised learning (SSL)** for multi- and hyperspectral imagery using the **Spectral Earth** dataset. We focus on **Masked Autoencoder (MAE)** pre-training to learn transferable spectral–spatial representations, and demonstrate how to adapt the pretrained model to downstream remote sensing tasks.

In this tutorial, you will explore:

- **Hyperspectral Data**
  - We use data from the **Spectral Earth** dataset, sourced from **EnMAP** (Environmental Mapping and Analysis Program).
  - The full Spectral Earth dataset can be retrieved via the **DLR Geoservices Portal**.

- **MAE Pre-training (Self-Supervised Learning)**
  - Implement and train a **Masked Autoencoder (MAE)** for hyperspectral SSL.

- **Downstream Tasks**
  - Fine-tune the pretrained model for **crop classification** using labels from **EuroCrops**.

---

## Code & Tools
This tutorial relies on the following libraries and resources:

- **Spectral Earth Repository** — pretrained models, data access utilities, and reference implementations  
- **PyTorch Lightning** — high-level training framework on top of PyTorch  
- **Lightly (SSL)** — self-supervised learning utilities for computer vision  
- **PyTorch** — core deep learning framework  

---

## Prerequisites
To follow along, you should have:

- Basic understanding of **machine learning** and **deep learning**
- Familiarity with **PyTorch**
- Some exposure to **computer vision** and **remote sensing** (helpful, not required)

---

## Table of Contents
1. Setup and Installation  
2. Understanding Hyperspectral Data  
3. Self-Supervised Pre-training with MAE  
4. Downstream Tasks: Fine-tuning for Crop Classification  
5. Visualizing Learned Representations  
6. Exercises
