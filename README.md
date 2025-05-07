
# GAN vs. cGAN on the Adult Dataset

![Image](https://github.com/user-attachments/assets/c1385121-393f-447a-84a2-30f0cac1f7e3)

This repository implements two generative models—a standard GAN and a conditional GAN (cGAN)—to synthesize realistic tabular data from the Adult dataset. Both models are built in PyTorch, trained on noise (plus labels for the cGAN), and evaluated via **detection** (indistinguishability) and **efficacy** (utility) metrics.

## Table of Contents

1. [Overview](#overview)  
2. [Data](#data)  
3. [Preprocessing](#preprocessing)  
4. [Model Architectures](#model-architectures)  
5. [Training](#training)  
6. [Evaluation](#evaluation)  
7. [Results](#results)  
8. [Conclusion & Next Steps](#conclusion--next-steps)  
9. [Authors](#authors)  
## Overview

Generative models can produce synthetic data that augment or replace real datasets. Here, a standard GAN learns to generate Adult‐income rows purely from noise, while a cGAN conditions on the “income” label to steer its outputs. We compare these two approaches on how well they mimic the true data distribution and how useful the synthetic data are for downstream classifiers.

## Data

### Dataset overview

- **Source:** Adult dataset (ARFF format; ~**48,000 examples**)  
- **Features:** 6 continuous (normalized) + 8 categorical (one-hot) + binary target income (≤50K / >50K)


Here’s a preview of the raw data format before any transformations:

![Image](https://github.com/user-attachments/assets/488cf32b-b769-493f-bb9b-5d98a0f85be8)


### Preprocessing

Each row was transformed into a uniform numeric vector by:

-  **Normalizing** all continuous features (e.g. age, hours-per-week) to the [0, 1] range.  
-  **One-hot encoding** each categorical feature (e.g. workclass, education, marital status, occupation, etc.).  


### Train/Test split

- 80 % train / 20 % test, _stratified_ on `income`
- Repeated 3× with different random seeds for robust averages  
After these steps, every example is a fixed-length vector (continuous + one-hot dims) ready for input into both the GAN and cGAN networks.

## Model architecture

### Generator (shared)

- **Input:**  
  • Noise vector (100-dim)  
  • [+ one-hot label vector for cGAN]

- **Layers:**  
  1. `Linear(100 → 256)` → LeakyReLU(0.2) → BatchNorm  
  2. `Linear(256 → 512)` → LeakyReLU(0.2) → BatchNorm  
  3. `Linear(512 → 1024)` → LeakyReLU(0.2) → BatchNorm  
  4. `Linear(1024 → output_dim)` → Sigmoid  


### Discriminator (shared)

- **Input:**  
  • Sample vector (features)  
  • [+ one-hot label vector for cGAN]

- **Layers:**  
  1. `Linear(input_dim → 512)` → LeakyReLU(0.2) → Dropout(0.3)  
  2. `Linear(512 → 256)` → LeakyReLU(0.2) → Dropout(0.3)  
  3. `Linear(256 → 128)` → LeakyReLU(0.2)  
  4. `Linear(128 → 1)` → Sigmoid 

> By concatenating the label vector to both the generator’s noise input and the discriminator’s input, a single codebase supports both GAN and cGAN modes without duplication.

## Training

- **Epochs:** 100  
- **Batch size:** 64  
- **Noise dimension:** 100  
- **Optimizer:** Adam (learning rate = 2×10⁻⁴)  
- **Loss function:** Binary Cross-Entropy for both generator and discriminator  
- **Logging:** Generator & discriminator losses tracked per epoch

## Evaluation

1. **Detection metric:**  
   - Form a 50/50 mix of real vs. synthetic samples  
   - Train a Random Forest with 4-fold cross-validation  
   - Report average AUC (lower ⇒ more indistinguishable)

2. **Efficacy metric:**  
   - Train RF on real training data → evaluate on real test set → AUC₁  
   - Train RF on synthetic data → evaluate on the same real test set → AUC₂  
   - Report ratio AUC₂ / AUC₁ (closer to 1 ⇒ synthetic data is a good substitute)

## Results

| Model | Detection AUC (↓) | Efficacy Ratio (↑) |
|:-----:|:-----------------:|:------------------:|
| **GAN**  | 1.000             | 0.58 – 0.63        |
| **cGAN** | 1.000             | 0.79 – 0.89        |

- **Detection:**  Both GAN and cGAN samples are perfectly distinguished from real data (AUC=1.0), indicating the generators did not fully capture the true distribution.  
- **Efficacy:**  cGAN outperforms the standard GAN (up to ~0.89 vs. ~0.63) by leveraging the label signal, but neither reaches parity with real data.

GAN: Detection & Efficacy Output:

![Image](https://github.com/user-attachments/assets/0c65453e-bbeb-4aa1-b8e5-3dd75feb92fb)

cGAN: Detection & Efficacy Output:

![Image](https://github.com/user-attachments/assets/97a10713-ec3a-43a4-ab4b-d9ed8d4c27f6)

## Conclusion

## Conclusion & Next Steps

Neither model produces entirely realistic synthetic data (detection AUC = 1.0), yet the cGAN’s conditioned outputs offer substantially higher downstream utility, demonstrating the power of label conditioning in generative tasks.
This project has deepened our understanding of how generative models learn complex tabular distributions, the challenges of mode collapse, and the impact of architectural and loss-function choices. 
Future work could mitigate mode collapse by adopting approaches such as Wasserstein GANs with gradient penalties


## Authors

- Roi Garber

- Nicole Kaplan
