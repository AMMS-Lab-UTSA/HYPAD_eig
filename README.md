# HYPAD_eig

This repository contains a Python implementation of two examples presented in the accompanying paper:  

**Efficient and Accurate Computation of Arbitrary-Order Eigenpair Sensitivities Using Hypercomplex Automatic Differentiation**

## Description
The script `DerivativesCantilever_Undamped.py`:
- Computes derivatives of eigenvalues and eigenvectors using hypercomplex numbers (via `pyoti`) and the residual-based differentiation method for the Undamped Cantilever Beam Problem
- Measures computational performance metrics for different derivative orders

The script `DerivativesCantilever_Underdamped.py`:
- Computes derivatives of eigenvalues and eigenvectors using hypercomplex numbers (via `pyoti`) and the residual-based differentiation method for the Clasically Underdamped Cantilever Beam Problem
- Measures computational performance metrics for different derivative orders

The script `DerivativesSSPlate_Undamped.py`:
- Computes derivatives of eigenvalues and eigenvectors using hypercomplex numbers (via `pyoti`) and the residual-based differentiation method for the Simply Supported Plate Problem
- Measures computational performance metrics for different derivative orders

The script `DerivativesSSPlate_Underdamped.py`:
- Computes derivatives of eigenvalues and eigenvectors using hypercomplex numbers (via `pyoti`) and the residual-based differentiation method for the Simply Supported Plate Problem
- Measures computational performance metrics for different derivative orders

The scripts also provides differentiation of the analytical solutions.

## Requirements
To run this code you will need:
- Python 3.x
- The following Python libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pyoti` (custom library required for hypercomplex arithmetic). Available "https://mauriaristi.github.io/otilib/"

## Usage
Example execution:
```bash
ipython Script.py
