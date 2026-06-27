# Skin Lesion Classification Modeling

A machine learning project that classifies dermoscopic skin lesion images into 7 diagnostic categories using the [DermaMNIST](https://medmnist.com/) dataset. Two approaches are compared: regularized Logistic Regression and a custom Multi-Layer Perceptron (MLP) built in PyTorch.

---

## Problem Statement

Skin cancer is one of the most common cancers worldwide, and early detection is critical. This project explores whether classical ML and simple neural network approaches can distinguish between 7 lesion types from 28×28 pixel images — and honestly examines where they fall short.

---

## Dataset

**DermaMNIST** — a subset of the HAM10000 dataset, resized to 28×28 RGB images.

| Split | Samples |
|-------|---------|
| Train | 7,007 |
| Validation | 1,003 |
| Test | 2,005 |

**Classes:**

| Label | Condition |
|-------|-----------|
| `akiec` | Actinic Keratoses |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `vasc` | Vascular Lesions |

> **Class imbalance note:** The dataset is heavily skewed — `nv` (Melanocytic Nevi) accounts for ~67% of all training samples. Both models address this using class-weighted loss functions.

---

## Approach

### Preprocessing
- Normalized pixel values from `[0, 255]` to `[0.0, 1.0]`
- Flattened 28×28×3 images into 2,352-dimensional vectors for compatibility with sklearn and the MLP input layer

### Model 1 — Logistic Regression (scikit-learn)
- `class_weight='balanced'` to handle class imbalance
- Compared two regularization strengths: `C=1.0` (complex boundaries) vs `C=0.01` (simple boundaries)
- Selected `C=0.01` based on improved melanoma recall on the validation set

### Model 2 — MLP Neural Network (PyTorch)
- Architecture: `Linear(2352 → 128) → ReLU → Linear(128 → 7)`
- Weighted `CrossEntropyLoss` using `compute_class_weight` from scikit-learn
- Optimizer: SGD, learning rate 0.01
- Trained for 50 epochs with batch size 64

---

## Results

| Model | Test Accuracy | Melanoma Recall |
|-------|--------------|-----------------|
| Logistic Regression (C=0.01) | 53.17% | 0.50 |
| MLP Neural Network | 44.34% | 0.42 |

The logistic regression outperformed the MLP on overall accuracy and minority class recall. The MLP showed unstable validation curves (see below), indicating sensitivity to the learning rate and plain SGD optimization.

### Confusion Matrices
<img width="1600" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/e5f9fe55-e38d-4450-9b28-987880391190" />

### MLP Training Curves
<img width="1400" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/2aa1d905-995a-46d2-95d4-198dbb33ec61" />


The high variance in the validation accuracy curve suggests the MLP would benefit from a lower learning rate, Adam optimizer, or early stopping.

---

## Key Takeaways

- **Class imbalance is the central challenge.** Without `class_weight='balanced'`, both models collapse to predicting `nv` for nearly everything.
- **Logistic regression is surprisingly competitive** on flattened image data at this scale — it trains faster and generalizes more reliably than the MLP here.
- **The MLP architecture is a known limitation.** Flattening image pixels destroys spatial structure. A CNN would be the appropriate next step for this type of data.
- **Recall on melanoma (`mel`) was prioritized** over raw accuracy, since false negatives carry higher clinical risk in a medical context.

---

## Tech Stack

- Python, NumPy, Matplotlib
- scikit-learn (Logistic Regression, metrics, class weighting)
- PyTorch (MLP, DataLoader, CrossEntropyLoss)
- MedMNIST

---

## Future Improvements

- [ ] Replace MLP with a CNN to leverage spatial features
- [ ] Use Adam optimizer with learning rate scheduling
- [ ] Apply image augmentation (flips, rotations) to improve generalization
- [ ] Experiment with transfer learning (e.g. pretrained ResNet on DermaMNIST)
- [ ] Perform cross-validation on the logistic regression
