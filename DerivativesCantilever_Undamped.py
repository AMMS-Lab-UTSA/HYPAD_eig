#!/usr/bin/env python
# coding: utf-8

### DerivativesCantilever_Undamped.py
# This script  computes eigenvalue and eigenvector derivatives 
# for a cantilever beam model using pyoti (hypercomplex arithmetic) and scipy sparse solvers.

import pyoti.sparse as oti
import pyoti.core as coti
import numpy as np
import scipy.sparse as spr
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.linalg import lu_factor, lu_solve
import time
import os

# Utility function to find nearest value index
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# Analytical solution helper function for gamma roots
def funcGamma(gamma):
    return np.cosh(gamma) * np.cos(gamma) + 1

# Analytical eigenvalue and mode shape solution for comparison
def eigValSol(x, E, L, ρ, b, d, gamma, alg=np):
    Iy = b * d ** 3 / 12.0
    Iz = d * b ** 3 / 12.0
    A = b * d
    lamda_y = gamma ** 4 * E * Iy / (ρ * A * L ** 4)
    lamda_z = gamma ** 4 * E * Iz / (ρ * A * L ** 4)
    beta = (x / L) * gamma
    term1 = alg.sinh(gamma) + alg.sin(gamma)
    term2 = alg.cosh(gamma) + alg.cos(gamma)
    term3 = alg.sinh(beta) - alg.sin(beta)
    term4 = alg.cosh(beta) - alg.cos(beta)
    phi = (1.0 / (ρ * A * L) ** 0.5) * (term4 * term1 / term1 - term2 * term3 / term1)
    return lamda_y, lamda_z, phi

# Tangent matrix construction for residual method
def TangentMatrix(Kmat, Mmat, lamda, Phi):
    tmp = np.dot(Mmat, Phi)
    T = np.zeros((Kmat.shape[0] + 1, Kmat.shape[0] + 1))
    T[:-1, :-1] = Kmat - lamda * Mmat
    T[:-1, -1] = -tmp
    T[-1, :-1] = -tmp
    return T

# Residual computation for current solution guess
def global_residual(Kmat, Mmat, lamda, Phi, alg=oti):
    res = oti.zeros((Kmat.shape[0] + 1, 1))
    res[:Kmat.shape[0], 0] = oti.dot(Kmat - lamda * Mmat, Phi)
    dot_tmp = oti.dot(Phi.T, oti.dot(Mmat, Phi))
    res[Kmat.shape[0], 0] = 0.5 - dot_tmp[0, 0] / 2.0
    return res

# Element stiffness matrix for beam element
def StiffnessMatrix(nodes, Emod, bs, ht):
    Iy = (bs * ht ** 3.0) / 12.0
    Iz = (bs ** 3.0 * ht) / 12.0
    Ar = bs * ht
    L = oti.sqrt((nodes[0, 0] - nodes[1, 0]) ** 2.0 + (nodes[0, 1] - nodes[1, 1]) ** 2.0)
    A = 12.0 * Emod * Iz / (L ** 3)
    B = 6.0 * Iz * Emod / (L ** 2)
    C = 12.0 * Emod * Iy / (L ** 3)
    D = 6.0 * Iy * Emod / (L ** 2)
    E = 4.0 * Iy * Emod / L
    F = 2.0 * Iy * Emod / L
    G = 4.0 * Iz * Emod / L
    H = 2.0 * Iz * Emod / L
    K = oti.array([
        [ A,  0,  0,  B, -A,  0,  0,  B],
        [ 0,  C, -D,  0,  0, -C, -D,  0],
        [ 0, -D,  E,  0,  0,  D,  F,  0],
        [ B,  0,  0,  G, -B,  0,  0,  H],
        [-A,  0,  0, -B,  A,  0,  0, -B],
        [ 0, -C,  D,  0,  0,  C,  D,  0],
        [ 0, -D,  F,  0,  0,  D,  E,  0],
        [ B,  0,  0,  H, -B,  0,  0,  G]
    ])
    return K

# Element mass matrix for beam element
def MassMatrix(nodes, rho, bs, ht):
    Iy = (bs * ht ** 3.0) / 12.0
    Iz = (bs ** 3.0 * ht) / 12.0
    Ar = bs * ht
    L = oti.sqrt((nodes[0, 0] - nodes[1, 0]) ** 2.0 + (nodes[0, 1] - nodes[1, 1]) ** 2.0)
    const = (rho * Ar * L / 420.0)
    A = const * 156.0
    B = const * 22.0 * L
    C = const * 54.0
    D = const * 13.0 * L
    E = const * 4.0 * L * L
    F = const * 3.0 * L * L
    M = oti.array([
        [ A,  0,  0,  B,  C,  0,  0, -D],
        [ 0,  A, -B,  0,  0,  C,  D,  0],
        [ 0, -B,  E,  0,  0, -D, -F,  0],
        [ B,  0,  0,  E,  D,  0,  0, -F],
        [ C,  0,  0,  D,  A,  0,  0, -B],
        [ 0,  C, -D,  0,  0,  A,  B,  0],
        [ 0,  D, -F,  0,  0,  B,  E,  0],
        [-D,  0,  0, -F, -B,  0,  0,  E]
    ])
    return M

# Assembly routine for global matrix
def Assembly(kelem, topo, Kglobal):
    DME = np.array([
        4 * (topo[1]) + 1, 4 * (topo[1]) + 2, 4 * (topo[1]) + 3, 4 * (topo[1]) + 4,
        4 * (topo[2]) + 1, 4 * (topo[2]) + 2, 4 * (topo[2]) + 3, 4 * (topo[2]) + 4
    ], dtype=int) - 1
    for CONT1 in range(8):
        for CONT2 in range(8):
            DMEi = int(DME[CONT1])
            DMEj = int(DME[CONT2])
            Kglobal[DMEi, DMEj] += kelem[CONT1, CONT2]



# Define geometry and material properties 
print('Started Step #1')
L = 5.0
maxOrder = 10
Emod = 68.9e9 
rho = 2770.0 
bs = 0.01 
ht = 0.015 
ne = 50
ndim = 2
nnode_el = 2

# User Defined Source of eigenvalues and eigenvectors
# (False) : Externally Obtained
# (True)  : Estimate Eigenvalues 
calc_eig=False

# Generate mesh: node coordinates and connectivity
nodes = oti.zeros((ne + 1, ndim + 1))
nodes[:, 0] = oti.array(np.linspace(0, ne, num=ne + 1))
nodes[:, 1] = oti.array(np.linspace(0.0, L, num=ne + 1))

topo = np.zeros((ne, 3), dtype=int)
topo[:, 0] = np.linspace(0, ne, num=ne, dtype=int)
topo[:, 1] = np.linspace(0, ne - 1, num=ne, dtype=int)
topo[:, 2] = np.linspace(1, ne, num=ne, dtype=int)

# Initialize global matrices
nod = oti.zeros((nnode_el, ndim))
Kglob = oti.zeros((4 * (ne + 1), 4 * (ne + 1)))
Mglob = oti.zeros((4 * (ne + 1), 4 * (ne + 1)))

# Assembly of global stiffness and mass matrices
for i in range(ne):
    ind = topo[i, :]
    for k in range(nnode_el):
        nod[k, :] = nodes[int(ind[k + 1]), 1:]
    K = StiffnessMatrix(nod, Emod, bs, ht)
    M = MassMatrix(nod, rho, bs, ht)
    Assembly(K, ind, Kglob)
    Assembly(M, ind, Mglob)


# Apply boundary conditions by imposing large diagonal terms (TGV penalty method)
TGV = 1.0e36
Kglob2 = Kglob.copy()
Mglob2 = Mglob.copy()
nfixed=4
for i in range(nfixed):
    Kglob2[i, i] = TGV
    Mglob2[i, i] = TGV

# Convert to CSR format for sparse solver

Kglob2csr = spr.csr_matrix(Kglob2.real)
Mglob2csr = spr.csr_matrix(Mglob2.real)

if calc_eig:
    neigs = 40
    eigval, eigvec = spla.eigs(Kglob2csr, neigs, Mglob2csr, which='SM')
else:
    # Load external eigenvalues and eigenvectors
    mat_contents = loadmat('data/Nominal_Eigenvalues_Undamped.mat')
    eigvec = mat_contents['V']
    eigval = mat_contents['W'][:, 0]

print('Finished Step #1')
# Compute derivatives for first eigenpair
neig = 6
# Number of Variales of interest
Na = 5

# Initialize Arrays to Store OTI version of eigenvalues and eigenvectors
Phi_res=oti.zeros((4 * (ne + 1), neig))
lambda_res=oti.zeros((1, neig))

print('Began Differentiation Loop')
for i in range(neig):

    lamda = eigval[i+nfixed].real # Skip four eigenvalues that correspond to fixed DOF
    Phi = oti.array(eigvec[:, i+nfixed].real)
    print(f'Started Step #2 for Eigenvalue {i+1}')

    # Tangent Matrix Estimation
    T = TangentMatrix(Kglob2.real, Mglob2.real, lamda, Phi.real.flatten())
    
    print(f'Finished Step #2 for Eigenvalue {i+1}')
    # Tangent Matrix Factorization
    c, low = lu_factor(T)

    # Initialize State Vector
    u = oti.zeros((Kglob.shape[0] + 1, 1))
    u[:Kglob.shape[0], 0] = Phi
    u[Kglob.shape[0], 0] = lamda

    # Loop for Derivatives Computation
    print(f'Started Step #3 for Eigenvalue {i+1}')

    # Definiton of Perturbations to Input Parameters
    Emod=68.9e9+oti.e(1,order=1)
    rho=2770.0+oti.e(2,order=1)
    bs=0.01+oti.e(3,order=1)
    ht=0.015+oti.e(4,order=1)

    nodes = oti.zeros((ne+1,ndim+1))
    nodes[:,0] = oti.array(np.linspace(0, ne, num=ne+1))

    nodes[:,1] = oti.array(np.linspace(0.0, L,num=ne+1))
    nodes[:,1] = nodes[:,1]+nodes[:,1]*oti.e(5,order=1)/L
    

    for ordi in range(1, maxOrder + 1):
        # Define Truncation Order
        Emod += 0*oti.e(1,order=ordi)
        rho += 0*oti.e(1,order=ordi)
        bs += 0*oti.e(1,order=ordi)
        ht += 0*oti.e(1,order=ordi)

        nodes[:,1] += 0*oti.e(1,order=ordi)
        nod = oti.zeros((ne+1,ndim))

        # Evaluate OTI Residual
        Kglob = oti.zeros((4*(ne+1),4*(ne+1)))
        Mglob = oti.zeros((4*(ne+1),4*(ne+1)))

        for conti in range(ne):
            ind=topo[conti,:]
            for k in range(nnode_el):
                nod[k,:] = nodes[int(ind[k+1]),1:]
            
            K=StiffnessMatrix(nod,Emod,bs,ht)
            M=MassMatrix(nod,rho,bs,ht)
            Assembly(K,ind,Kglob)
            Assembly(M,ind,Mglob)

        Kglob2 = Kglob.copy()
        Mglob2 = Mglob.copy()
        nfixed=4
        for conti in range(nfixed):
            Kglob2[conti, conti] = TGV
            Mglob2[conti, conti] = TGV



        Phi = u[:Kglob.shape[0], 0] + 0 * oti.e(1, order=ordi)
        lamda = u[Kglob.shape[0], 0] + 0 * oti.e(1, order=ordi)
        u[Kglob.shape[0], 0] = lamda

        # Evaluation of Global Residual
        res = global_residual(Kglob2, Mglob2, lamda, Phi)
        # Extraction of OTI coefficients to form Matrix R
        rhsi = oti.get_order_im_array(ordi, res)
        # Solution of Linear Systems
        u_i = lu_solve((c, low), -rhsi)
        # Update of state vector with derivatives
        oti.set_order_im_from_array(ordi, u_i, u)

    Phi_res[:,i] = u[:Kglob.shape[0],0]
    lambda_res[0,i] = u[Kglob.shape[0],0]

    print(f'Finished Step #3 for Eigenvalue {i+1}')

# Extraction of sensitivities of interest
# ---------------------------------------
print(f'Started Step #4')
# Nominal (0th-order) eigenvalues:
eigvals = lambda_res.real

# Example: Extract 4th-order eigenvalue derivative w.r.t variable 2 (ρ - mass density)
der1_val = lambda_res.get_deriv([[2, 4]])

# Example: Extract 4th-order eigenvector derivative w.r.t variable 2 (ρ - mass density)
der1_vec = Phi_res.get_deriv([[2, 4]])

# Example: Extract 5th-order eigenvalue mixed derivative w.r.t all 5 design variables
der2_val = lambda_res.get_deriv([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]])


# General extraction explanation:
# - The argument to `get_deriv()` is a list of lists:
#   Each sublist: [variable_index, order_of_derivative]
#   Example: [2, 4] means 4th derivative w.r.t design variable 2.
# - Higher-order mixed derivatives follow the same syntax:
#   Example: [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]] for 1st derivatives w.r.t all five variables.

# Loop to collect all derivatives up to maxOrder into an array `der_eigvals`

der_eigvals=np.zeros((neig,coti.ndir_total(Na,maxOrder)))
for i in range(neig):
    der_eigvals_i=lambda_res[0,i].real
    for ordi in range(1, maxOrder + 1):
        num = oti.array([lambda_res[0,i]])
        
        # Extract coefficients for derivative order `ordi` from OTI representation
        cnum = oti.get_order_im_array(ordi, num)

        # Append this derivative order to the output array
        der_eigvals_i = np.append(der_eigvals_i, cnum)
    der_eigvals[i,:]=der_eigvals_i


# Save all extracted derivatives to MATLAB .mat file for postprocessing
mdic = {"der_eigvals": der_eigvals}
os.makedirs('results', exist_ok=True)
savemat("results/DerivativesEigenvaluesUndamped.mat", mdic)
