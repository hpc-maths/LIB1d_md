# LIB1d_md

**Multi-domain time integration method with adaptive, implicit/explicit, and high-order coupling demonstrated for 1D lithium-ion batteries half-cell simulations at the microscale.**

This repository provides Jupyter notebooks that can reproduce the main results presented in [1].

## Getting Started

If you don’t have Anaconda or Miniconda installed, please install it first from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).

Then, create and activate the conda environment using the provided `environment.yml` file:

```bash
conda env create -f conda/environment.yml
conda activate lib1d_md
```

You can then launch the notebooks:
```bash
jupyter notebook 
```

[1] A. Asad, R. de Loubens, L. François, and M. Massot,  
*High-order adaptive multi-domain time integration scheme for microscale lithium-ion batteries simulations*,  
SMAI Journal of Computational Mathematics, 2024 (*article in revision*).  
Available at: [https://arxiv.org/abs/2310.06573](https://arxiv.org/abs/2310.06573).

