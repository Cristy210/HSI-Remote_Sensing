
<div align="center">
<h1> Hyperspectral Image Analysis on Remote sensing datasets </h1>
</div>

<!-- ## HSI-Remote_Sensing -->

**Hyperspectral Imaging** is a powerful technique that captures and analyzes a wide spectrum of light across hundreds of narrow bands. This technology allows for detailed material identification, pattern recognition, and classification across diverse fields, including environmental monitoring, medical imaging, agriculture, art conservation, and more.

## 🔧 Language and Tools

This repository provides subspace-based clustering implementations in **Python**, **Julia**, and **MATLAB** offering flexibility for experimentation and optimization. 

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow.svg)]()
[![Julia](https://img.shields.io/badge/Julia-1.10+-green.svg)]()
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023+-orange.svg)]()

## 🌍 Datasets

This project currently works with these public hyperspectral remote sensing datasets. 

- **Pavia Centre and Pavia University Scene** - Captured by the ROSIS sensor over Pavia, Italy. 

- **Salinas** - Acquired by the AVIRIS sensor over Salinas Valley, California.

- **Onera Satellite Change Detection** - Images acquired by sentinel-2 satellites between 2015 and 2018 with locations all over the world, in Brazil, USA, Europe, Middle-East, and Asia. 

## 📂 Repository Structure

```bash
HSI-Remote_Sensing/
├── KSS/                    # Python implementation using MLX
│   ├── kss_mlx.py          # Core KSS class definition
│   ├── __init__.py         # Module init file
│
├── KSS_MATLAB/             # MATLAB implementation
│   ├── kss.m               # Core function
│   ├── kss_rs.m            # Remote sensing dataset demo
│ 
├── KSS_Julia/              # Julia Implementation
│   ├── kss.jl              # Core kss function
│
├── TSC_Python/             # Python Implementation
│    ├── tsc.py             # Core TSC Implementation
│    ├── tsc_mlx.py         # TSC Implementation using MLX framework
│    ├── __init__.py        # Module init file 
│
├── notebooks/              # Data loading and visualization notebooks
├── CITATIONS.md            # Dataset and paper references
├── env.yml                 # Conda environment configuration
└── README.md
```
> 📖 See [CITATIONS.md](CITATIONS.md) for links and references to datasets and algorithms used in this project.
