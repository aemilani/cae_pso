# CAE-PSO

A Convolutional Autoencoder trained using the PSO (Particle Swarm Optimization) algorithm for extracting latent monotonic factors from multivariate time series.

Here we provide information about:
1. Introduction
2. Installation
3. Basic usage
4. Citation

## 1. Introduction

`cae-pso` is a Python package designed for unsupervised feature extraction from complex multivariate time series. By integrating a Convolutional Autoencoder (CAE) with Particle Swarm Optimization (PSO), this model is optimized to construct latent monotonic factors. This approach is highly effective for applications such as generating robust health indicators for predictive maintenance, condition monitoring, and degradation tracking (e.g., bearing health assessment) in wind energy and aerospace systems.

The package implements the hybrid training methodology proposed in the following research:
> **A hybrid Convolutional Autoencoder training algorithm for unsupervised bearing health indicator construction**
> *Engineering Applications of Artificial Intelligence*, Volume 139, 2025.
> [Read the full paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S095219762401635X)

In this work, the model is applied to extract the monotonic factor describing the degradation in bearings, i.e.,
Health Indicators (HIs). Leveraging PSO for training the CAE architecture enables maximising the global monotonicity of
the extracted factor.

## 2. Installation

You can install the latest release of the package directly from PyPI.

```bash
pip install cae-pso
```

## 3. Basic usage

Below is a quick-start example demonstrating how to initialize the `CAE` model, prepare your data with the expected dimensional shapes, run the PSO training, and extract the monotonic health indicator (HI).

```python
import numpy as np
from cae_pso import CAE

# 1. Prepare your multivariate time series data
# The model expects a list of arrays for training (e.g., multiple run-to-failure trajectories).
# Each array MUST have the shape: (1, total_time_steps, n_features, 1)
total_time_steps = 1000
n_features = 4
time_window = 30

# Creating a single dummy run-to-failure (RTF) trajectory
rtf_trajectory = np.random.rand(1, total_time_steps, n_features, 1)
train_data = [rtf_trajectory] # Add more trajectories to this list as needed

# 2. Initialize the model
# Define the time window (sequence length) and the number of features
model = CAE(
    time_w=time_window, 
    n_features=n_features,
    n_filters=16,       # Default: 16 filters
    activation='elu'    # Default: Exponential Linear Unit
)

# 3. Train the model using PSO
# The train method utilizes multiprocessing and logs the generation stats
history = model.train(
    data=train_data, 
    n_gen=50,                   # Number of generations
    pop_size=20,                # Particle swarm population size
    log_filepath='./logs/'      # Ensure this directory exists or update the path
)

# Access training history if needed
# print("Best Fitness:", history.maxs[-1])

# 4. Extract the Health Indicator (HI)
# Pass a single trajectory to get_hi() to extract the smoothened, monotonic trend
# Output will be a 1D array representing the health indicator over time
health_indicator = model.get_hi(rtf_trajectory)

print(f"Extracted HI shape: {health_indicator.shape}")
```

## 4. Citation

If you use this package in your research or work, please cite the original paper:

```bibtex
@article{Milani2025,
  title = {A hybrid Convolutional Autoencoder training algorithm for unsupervised bearing health indicator construction},
  volume = {139},
  ISSN = {0952-1976},
  url = {http://dx.doi.org/10.1016/j.engappai.2024.109477},
  DOI = {10.1016/j.engappai.2024.109477},
  journal = {Engineering Applications of Artificial Intelligence},
  publisher = {Elsevier BV},
  author = {Milani,  Ali Eftekhari and Zappalá,  Donatella and Watson,  Simon J.},
  year = {2025},
  month = Jan,
  pages = {109477}
}
```