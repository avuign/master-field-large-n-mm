# Regularized Master-Field Approximation for Large-N Matrix Models

A PyTorch implementation of the Regularized Master-Field Approximation (RMA) for the large-$N$ one-matrix model, replicating the results of [Maeta (2026)](https://arxiv.org/abs/2605.10720).

## Overview

In the large-$N$ limit of matrix models, correlation functions factorize and the theory admits a classical description in terms of a single infinite-dimensional matrix — the **master field**. This project implements the RMA: the master field is regularized to a finite $M \times M$ matrix whose eigenvalues are optimized to satisfy the loop equations (Schwinger-Dyson equations) as accurately as possible.

The model is the Hermitian one-matrix model with quartic potential:

$$S = N \,\mathrm{Tr}\left(\frac{1}{2}\phi^2 + g\phi^4\right)$$

The primary observables are the single-trace moments $w_{2k} = \frac{1}{N}\langle \mathrm{Tr}\, \phi^{2k} \rangle$, which at large $N$ satisfy an infinite tower of loop equations used as the optimization objective.

## Method

- **Parameters**: $M/2$ positive eigenvalues $\{\lambda_i\}$, with the full spectrum enforced to be symmetric ($\pm\lambda_i$) by construction, exploiting the $\mathbb{Z}_2$ symmetry of the potential
- **Moments**: computed as $w_n = \frac{1}{M}\sum_i \lambda_i^n$ directly from the eigenvalues
- **Loss**: sum of squared loop equation residuals, normalized by the exact moment scale $(a^2/4)^{n/2}$ to keep all terms $O(1)$
- **Optimizer**: Adam

The exact solution for $g > 0$ is known analytically via the resolvent method.

## Results

The RMA accurately reproduces the exact moments $w_{2k}$ for the Hermitian one-matrix model. Accuracy improves with the master field size $M$ and degrades for higher-order moments near the truncation.

## Repository Structure

```
master-field-large-n-mm/
└── 1-mm.ipynb    # full implementation and experiments
```

## Requirements

```
torch
matplotlib
numpy
```

Install with:

```bash
pip install torch matplotlib numpy
```

## Usage

Open the notebook and run all cells:

```bash
jupyter lab 1-mm.ipynb
```

Key hyperparameters at the top of the notebook:

| Parameter | Description |
|-----------|-------------|
| `M` | Master field size (number of eigenvalues) |
| `K` | Number of loop equations imposed (use `K = 2*M`) |
| `g` | Quartic coupling constant |
| `lr` | Adam learning rate |
| `num_epochs` | Number of optimization steps |

## References

- Maeta, R. (2026). *Regularized Master-Field Approximation for Large-N Reduced Matrix Models*. [arXiv:2605.10720](https://arxiv.org/abs/2605.10720)
