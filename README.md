# HYPAD_eig

Developed by [Juan C. Velasquez-Gonzalez](https://orcid.org/0000-0003-2442-437X) and collaborators. 

This repository contains a Python implementation of two examples presented in the accompanying paper:  

**Efficient and Accurate Computation of Arbitrary-Order Eigenpair Sensitivities Using Hypercomplex Automatic Differentiation**

## Description
The script `DerivativesCantilever_Undamped.py`:
- Computes derivatives of eigenvalues and eigenvectors using hypercomplex numbers (via `pyoti`) and the residual-based differentiation method for the Undamped Cantilever Beam Problem

The script `DerivativesCantilever_Underdamped.py`:
- Computes derivatives of eigenvalues and eigenvectors using hypercomplex numbers (via `pyoti`) and the residual-based differentiation method for the Clasically Underdamped Cantilever Beam Problem


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

## Citation
If you use this framework/dataset, build on or find our research useful for your work please cite as, 
```
@misc{Velasquez-Gonzalez2025HYPAD_eig,
  author       = {Velasquez-Gonzalez, Juan C. and Aristizabal, Mauricio and Navarro, Juan and Millwater, Harry and Restrepo, David},
  title        = {HYPAD_eig},
  year         = {2025},
  month        = {August},
  day          = {07},
  version      = {0.1},
  howpublished = {\url{https://github.com/AMMS-Lab-UTSA/HYPAD_eig}},
  note         = {GitHub repository}
}
```
