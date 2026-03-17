# Semi-Supervised Medical Image Classification

> A practical pipeline comparing unsupervised clustering and semi-supervised learning to classify medical images with limited labels.

## Project Overview

This project explores how to classify grayscale medical images when only a small portion of the dataset is labeled. It compares two approaches:

- **Unsupervised clustering** on CNN embeddings (PCA + KMeans/DBSCAN)
- **Semi-supervised learning** using label propagation and pseudo-labelling

The work is framed around a realistic constraint: manual annotation is expensive, so the goal is to improve performance while minimizing labeling effort.

## Key Results (from the presentation)

- Unsupervised clustering shows limited separation (visual structure is weak; ARI ~ 0.11).
- Semi-supervised learning provides strong performance (accuracy ~ 98%, ROC curves strong).
- Label propagation alone can be unstable (class imbalance after propagation).
- Pseudo-labelling with confidence thresholds yields more balanced labels and better results.

## What This Project Is / Is Not

**This project is:**

- A reproducible pipeline to compare unsupervised vs semi-supervised classification
- A practical baseline using pretrained CNN embeddings
- A foundation for scaling weak-labeling strategies

**This project is not:**

- A clinically validated medical system
- A production deployment
- A segmentation model (classification only)

## Repository Structure

```
notebooks/            # Main analysis notebooks
src/                  # Feature extraction and semi-supervised utilities
models/               # Model artifacts (if generated locally)
data/                 # Labeled/unlabeled image data
Dockerfile            # Local dev image
Dockerfile            # Local dev image
Makefile              # Convenience commands
```

## Notebooks

- `notebooks/01_Extraction_Clustering.ipynb` - CNN feature extraction + clustering
- `notebooks/02_Semi_Supervised_Approach.ipynb` - Label propagation + pseudo-labelling

## Methods Summary

### Feature Extraction
- Pretrained CNN backbone (e.g., ResNet18 without final layer)
- Image preprocessing: resize, normalization, standardization
- Embeddings saved as `.npy` for reuse

### Unsupervised Clustering
- PCA / t-SNE for visualization
- KMeans and DBSCAN for grouping
- Visual inspection + clustering metrics

### Semi-Supervised Learning
- Label propagation (rbf/knn kernels)
- Pseudo-labelling with confidence threshold
- Supervised classifier trained on enriched labels

## Scaling Recommendation (from the presentation)

Goal: label ~4M images with ~5,000 EUR budget.

- Use pseudo-labelling with a pretrained lightweight model
- Run inference on spot GPU instances
- Estimated cost:
  - Inference: ~4,000 EUR (0.001 EUR / image)
  - Infra + monitoring: ~1,000 EUR

Feasible under conditions: stable pipeline, parallelization, and controlled GPU costs.

## Local Development

```bash
docker compose --profile cpu build
make jupyter
```

## License

MIT License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software, provided that the original copyright notice and license terms are included.
