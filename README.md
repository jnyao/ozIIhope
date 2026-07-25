# ozHOPE

This repository provides the artifact for **High-Order-Preserving Acceleration of a Shallow-Water Dynamical Core Using Tensor Units**.

It implements an Ozaki-based Python/PyTorch extension for accelerating HOPE convolution computations with low-precision Tensor Cores while preserving the numerical properties of the high-order finite-volume shallow-water dynamical core.

## Requirements

Tested environment:

- NVIDIA A40 / RTX A6000 GPU
- CUDA 11.8
- cuDNN 9.1.0
- CMake 3.24.3
- Python 3.13.5
- PyTorch 2.6.0
- cuDNN-frontend Python interface 1.12.0

## Setup

Set the cuDNN library path in `CMakeLists.txt`, then compile the Ozaki library and its Python extension.

After compilation, Ozaki-based convolution can be enabled or disabled in HOPE through the `prec_mode` switch.

## Running

The artifact workflow is:

1. Generate ghost interpolation matrices during the first run.
2. Run HOPE simulations with selected benchmark cases, resolutions, and convergence orders.
3. Collect diagnostic variables and compare against FP64 reference results.

Benchmark cases:

- Steady-state geostrophic flow
- Rossby-Haurwitz wave
- Perturbed jet flow

## Expected Results

The optimized cuDNN-based Ozaki implementation should outperform the FP64 baseline while preserving the expected convergence behavior or producing results consistent with FP64 references, depending on the benchmark case.
