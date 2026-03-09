# Matrix Completion for Recommender Systems

This project explores convex optimization techniques applied to the **Matrix Completion** problem, a core task in Recommender Systems. The goal is to predict missing user ratings by exploiting the low-rank structure of user-item interaction matrices.

##  Overview

The repository implements and compares three main optimization strategies to minimize the Mean Squared Error (MSE) under nuclear norm or rank constraints:

1. **Frank-Wolfe (FW)**: A projection-free optimization algorithm using a Singular Value Decomposition (SVD) based Linear Minimization Oracle (LMO).
2. **Pairwise Frank-Wolfe (PFW)**: An improved version of the FW algorithm that accelerates convergence. This implementation features an efficient **Active Set management** using CRC32 hashing for atom identification.
3. **Projected Gradient Descent (PGD)**: A standard gradient-based approach using hard rank-constraint projections via truncated SVD.

##  Datasets

The project includes automated scripts to download and preprocess the following benchmarks:

* **MovieLens 100k**: The classic movie recommendation dataset.
* **Amazon Office Products**: User reviews and ratings from Amazon's office product category.
* **SweetRS**: A specific dataset for recommendation analysis.

##  Tech Stack

* **JAX**: For high-performance numerical computing, GPU acceleration, and Just-In-Time (JIT) compilation.
* **NumPy & Pandas**: Data manipulation and matrix handling.
* **Matplotlib & Seaborn**: Performance visualization and convergence analysis.
* **Scikit-learn**: Evaluation metrics (RMSE).

##  Key Features

* **Custom Line Search**: Implementation of Armijo Backtracking and Exact Line Search for optimal step-size selection.
* **Automated Pipeline**: Integrated hyperparameter tuning and model selection.
* **Performance Benchmarking**: Comparative analysis focusing on RMSE accuracy, computational efficiency (wall-clock time), and model complexity (matrix rank).

##  Getting Started

### Prerequisites

Ensure JAX is installed with appropriate CUDA support for GPU execution:

```bash
pip install numpy pandas tqdm matplotlib scikit-learn jax[cuda12] seaborn

```

### Usage

1. Open `recommender_system_project.ipynb` in Google Colab or a local Jupyter environment.
2. Run the initialization cells to load dependencies and define optimization oracles.
3. Execute the `run_pipeline` function specifying the desired dataset to start training and evaluation.

---

*Developed as part of a research project on optimization algorithms for machine learning.*
