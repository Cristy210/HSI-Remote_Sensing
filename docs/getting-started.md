# Getting Started

This page walks you through setting up the repository and running your first clustering experiment.

## 1. Clone the repository

```bash
git clone https://github.com/Cristy210/HSI-Remote_Sensing.git
cd HSI-Remote_Sensing
```

## 2. Install dependencies (Python)
This project uses a Conda environment for reproducibility and to support MLX (Apple Silicon GPU acceleration).
Create and activate the environment
```bash
conda env create -f env.yml
conda activate HSIRS
```
## 3. Run a Test Sample (KSS Python)
```
from KSS.kss_mlx import KSS
import numpy as np

# Create random data matrix (D x N)
X = np.random.randn(100, 500)

kss = KSS(K=3, r=2, max_iter=10)
labels = kss.fit_predict(X)

print(labels)
```
