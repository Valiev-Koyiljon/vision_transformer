# Vision Transformer (ViT)

A PyTorch implementation of Vision Transformer for image classification on the CIFAR-10 dataset.

## Overview

This project implements the Vision Transformer architecture as described in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020). The model divides images into fixed-size patches, linearly embeds them, and processes them with a standard Transformer encoder for classification tasks.

## Features

* Pure PyTorch Implementation: Clean, modular code for the Vision Transformer architecture
* CIFAR-10 Classification: Trained and tested on the CIFAR-10 dataset (10 object classes)
* Configurable Architecture: Easily adjustable hyperparameters for patch size, embedding dimensions, number of heads, and layers
* Training & Evaluation: Complete training pipeline with validation and test evaluation

## Requirements

```
python>=3.8
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Valiev-Koyiljon/vision_transformer.git
cd vision_transformer
pip install -r requirements.txt
```

## Model Architecture

The Vision Transformer consists of:

1. **Patch Embedding**: Splits input images into non-overlapping patches and linearly projects them
2. **Positional Encoding**: Adds learnable positional embeddings to patch embeddings
3. **Transformer Encoder**: Stack of multi-head self-attention and feed-forward layers
4. **Classification Head**: Linear layer for predicting class labels

## Usage

Open and run `vit.ipynb` in Jupyter Notebook or JupyterLab. The notebook contains the complete implementation, training, and evaluation pipeline for the Vision Transformer on CIFAR-10.

## Dataset

The model is trained and evaluated on CIFAR-10, which contains:

* 60,000 RGB images (32×32 pixels)
* 10 object classes
* 50,000 training images
* 10,000 test images

The dataset is automatically downloaded during first run.

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)