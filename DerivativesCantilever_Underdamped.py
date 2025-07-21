#!/usr/bin/env python
# coding: utf-8

# In[1]:

### VERSION OF THE CODE WITH NO CORNERS. THIS CODE APPLIES BLOCH PERIODICITY 
### AND COMPUTES DERIVATIVES

import pyoti.sparse as oti
import pyoti.core as coti
import numpy as np
# import scipy #as scy
import scipy.sparse as spr
from scipy.io import savemat,loadmat
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.linalg import cho_factor, cho_solve,lu_factor, lu_solve
import os
import time
import scipy as sci
import sys
# import warnings
# warnings.filterwarnings("ignore")


def assemble_vec(vg, ve, dme):
    """Add *ve* (8×1) into the global vector *vg* through *dme* DOF map."""
    for a in range(8):
        vg[int(dme[a]), 0] += ve[a, 0]


def assemble_mat(Mg, Me, dme):
    """Symmetric assembly of an 8×8 element matrix into dense *Mg*."""
    for a in range(8):
        ia = int(dme[a])
        for b in range(8):
            ib = int(dme[b])
            Mg[ia, ib] += Me[a, b]

# ---------------------------------------------------------------------------
#  Element‑level residual and tangent evaluation
# ---------------------------------------------------------------------------
def gather_rows(vec, idx):
    return oti.concatenate([vec[i] for i in idx], axis=0)   # (8×1) OTI column

def zero_items_1(M,i):
    M[i,1] = 0
    M[i,2] = 0
    M[i,5] = 0
    M[i,6] = 0

def zero_items_2(M,i):
    M[i,0] = 0
    M[i,3] = 0
    M[i,4] = 0
    M[i,7] = 0

def zero_items(M):
    zero_items_1(M,0)
    zero_items_2(M,1)
    zero_items_2(M,2)
    zero_items_1(M,3)
    zero_items_1(M,4)
    zero_items_2(M,5)
    zero_items_2(M,6)
    zero_items_1(M,7)
    
def global_residual(Kmat, Mmat, Cmat, lamda_re, Phi_Re, lamda_imag, Phi_imag, alg=oti):


    # ---------- 1.  Pre-compute common blocks ----------
    A_re = Kmat + Cmat * lamda_re - Mmat * (lamda_imag**2 - lamda_re**2)
    B_re = Cmat * lamda_imag + Mmat*(2.0  * lamda_imag * lamda_re)

    v_re = alg.dot(A_re, Phi_Re) - alg.dot(B_re, Phi_imag)

    
    v_im = alg.dot(A_re, Phi_imag) + alg.dot(B_re, Phi_Re)


    Ce_M_re = Cmat + 2.0 * Mmat * lamda_re          # symmetric
    Me_imag = Mmat *(2.0 *  lamda_imag )             # symmetric

    # Precompute reused matrix-vector products
    Ce_M_re_vecR = oti.dot(Ce_M_re, Phi_Re)
    Ce_M_re_vecI = oti.dot(Ce_M_re, Phi_imag)
    Me_imag_vecR = oti.dot(Me_imag, Phi_Re)
    Me_imag_vecI = oti.dot(Me_imag, Phi_imag)

    Phi_Re_T = Phi_Re.T
    Phi_imag_T = Phi_imag.T



    # Exploit symmetry: xᵀAy = yᵀAx → avoid recomputation
    tmpR = (
        oti.dot(Phi_Re_T, Ce_M_re_vecR)
        - oti.dot(Phi_imag_T, Ce_M_re_vecI)
        - 2 * oti.dot(Phi_Re_T, Me_imag_vecI)  # use symmetry instead of two terms
    )
    stime3 = time.time()
    tmpI = (
        2 * oti.dot(Phi_Re_T, Ce_M_re_vecI)    # xᵀAy + yᵀAx
        - oti.dot(Phi_imag_T, Me_imag_vecI)
        + oti.dot(Phi_Re_T, Me_imag_vecR)
    )



    # ---------- 4.  Assemble residual vectors ----------
    n = Kmat.shape[0]
    res_re = alg.zeros((n + 1, 1))
    res_im = alg.zeros((n + 1, 1))


    res_re[:(Kmat.shape[0]),0] = v_re
    res_re[Kmat.shape[0],0] = tmpR

    res_im[:(Kmat.shape[0]),0] = v_im

    res_im[Kmat.shape[0],0] = tmpI

    
    return res_re,res_im


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


## Analytical Solution
def funcGamma(gamma):
    return np.cosh(gamma)*np.cos(gamma)+1

def eigValSol(x,E,L,ρ,b,d,gamma,alpha,beta,alg=np):
    Iy = b*d**3/12.0
    Iz = d*b**3/12.0
    A  = b*d
    lamda_y = alg.sqrt(gamma**4 * E * Iy /(ρ*A*L**4))
    lamda_z = alg.sqrt(gamma**4 * E * Iz /(ρ*A*L**4))

    ## Damped Natural Frequencies
    xii=0.5*(alpha/lamda_y+beta*lamda_y)
    lamda_y=lamda_y*alg.sqrt(1.-xii**2.0)

    xii=0.5*(alpha/lamda_z+beta*lamda_z)
    lamda_z=lamda_z*alg.sqrt(1.-xii**2.0)


    
    term1 = alg.sinh(gamma)+alg.sin(gamma)
    term2 = alg.cosh(gamma)+alg.cos(gamma)
    
    beta = (x/L)*gamma ## Care with 
    
    term3 = alg.sinh(beta)-alg.sin(beta)
    term4 = alg.cosh(beta)-alg.cos(beta)
    
    phi     = (1.0/(ρ*A*L)**0.5)*(term4*term1/term1 - term2*term3/term1)
    return lamda_y,lamda_z,phi




## Functions and Subroutines
def TangentMatrix(Kmat, Mmat, Cmat, lamda_re,lamda_im, Phi_re,Phi_im ):
    """
    Generates the tangent matrix for the residual method to be applied.
        K: Stiffness matrix (dense for now)
        M: Mass matrix (dense for now)
        lamda: Eigenvalue
        Phi: Eigenvector 

    
    """
    Phi=Phi_re+Phi_im*1j
    lamda=lamda_re+lamda_im*1j

    
    T = np.zeros((Kmat.shape[0]+1,Kmat.shape[0]+1),dtype=complex)

    tmp = np.dot( 2.0*Mmat*lamda+Cmat, Phi)

    T[:-1,:-1] = Mmat*lamda**2.0+Cmat*lamda+Kmat
    T[:-1,-1]  =  tmp
    
    T[-1,:-1]  =  tmp

    tmp2 = np.dot( Mmat, Phi)

    # tmp3 = np.dot( Phi,tmp2.flatten())
    # tmp3 = np.dot(Phi, tmp2.flatten().reshape(-1, 1))
    tmp3 = np.dot(Phi.T, tmp2.T)

    T[-1,-1]   = 2.0*tmp3



    # T = spr.csc_matrix(T)
    
    return T



def StiffnessMatrix(nodes,Emod,bs,ht):
    
    ## nodes: 2x2 array with coordinates of the two nodes.The columns are the coordinate x y.Rows: Node
    ## Emod: Youngs Modulus
    ## bs: Base of the rectangular cross section
    ## ht: Heigth of the rectangular cross section
    Iy=(bs*ht**3.0)/12.0
    Iz=(bs**3.0*ht)/12.0
    Ar=bs*ht
    
    L=oti.sqrt((nodes[0,0]-nodes[1,0])**2.0+(nodes[0,1]-nodes[1,1])**2.0)
    

    A=12.0*Emod*Iz/(L*L*L)
    B= 6.0*Iz*Emod/(L*L)
    C=12.0*Emod*Iy/(L*L*L)
    D= 6.0*Iy*Emod/(L*L)
    E= 4.0*Iy*Emod/L
    F= 2.0*Iy*Emod/L
    G= 4.0*Iz*Emod/L
    H= 2.0*Iz*Emod/L
    
    
    
    K=oti.array([[ A,  0,  0,  B, -A,  0,  0,  B],
                 [ 0,  C, -D,  0,  0, -C, -D,  0],
                 [ 0, -D,  E,  0,  0,  D,  F,  0],
                 [ B,  0,  0,  G, -B,  0,  0,  H],
                 [-A,  0,  0, -B,  A,  0,  0, -B],
                 [ 0, -C,  D,  0,  0,  C,  D,  0],
                 [ 0, -D,  F,  0,  0,  D,  E,  0],
                 [ B,  0,  0,  H, -B,  0,  0,  G]])
    
    return K


def MassMatrix(nodes,rho,bs,ht):
    
    ## nodes: 2x2 array with coordinates of the two nodes.The columns are the coordinate x y.Rows: Node
    ## Emod: Youngs Modulus
    ## bs: Base of the rectangular cross section
    ## ht: Heigth of the rectangular cross section
    Iy=(bs*ht**3.0)/12.0
    Iz=(bs**3.0*ht)/12.0
    Ar=bs*ht
    
    L=oti.sqrt((nodes[0,0]-nodes[1,0])**2.0+(nodes[0,1]-nodes[1,1])**2.0)
    
    const = (rho*Ar*L/420.0)
    A = const*156.0
    B = const*22.0*L
    C = const*54.0
    D = const*13.0*L
    E = const*4.0*L*L
    F = const*3.0*L*L
    
    M=oti.array([[ A,  0,  0,  B,  C,  0,  0, -D],
                 [ 0,  A, -B,  0,  0,  C,  D,  0],
                 [ 0, -B,  E,  0,  0, -D, -F,  0],
                 [ B,  0,  0,  E,  D,  0,  0, -F],
                 [ C,  0,  0,  D,  A,  0,  0, -B],
                 [ 0,  C, -D,  0,  0,  A,  B,  0],
                 [ 0,  D, -F,  0,  0,  B,  E,  0],
                 [-D,  0,  0, -F, -B,  0,  0,  E]])
    
    return M
def Assembly(kelem,topo,Kglobal):
    DME=np.array([4*(topo[1])+1, 4*(topo[1])+2, 4*(topo[1])+3,
                  4*(topo[1])+4, 4*(topo[2])+1, 4*(topo[2])+2,
                  4*(topo[2])+3, 4*(topo[2])+4],dtype=int)-1
#     print(DME)
    for CONT1 in range(8):
        for CONT2 in range(8):
            DMEi = int(DME[CONT1])
            DMEj = int(DME[CONT2])
            Kglobal[DMEi,DMEj]=Kglobal[DMEi,DMEj]+kelem[CONT1,CONT2]



# Define geometry and material properties (with perturbations for derivative computation)
print('Started Step #1')

L=5.0
maxOrder = 10
Emod=68.9e9
rho=2770.0
bs=0.01
ht=0.015

alpha_r  = 1.0E-1
beta_r   = 1.0E-2

ne=50
ndof = 2
nnode_el = 2

# User Defined Source of eigenvalues and eigenvectors
# (False) : Externally Obtained
# (True)  : Estimate Eigenvalues 
calc_eig=False

# Generate mesh: node coordinates and connectivity
nodes = oti.zeros((ne+1,ndof+1))
nodes[:,0] = oti.array(np.linspace(0, ne, num=ne+1))
nodes[:,1] = oti.array(np.linspace(0.0, L,num=ne+1))


topo=np.zeros((ne,3),dtype=int)
topo[:,0] = np.linspace(0,  ne,num=ne,dtype=int)
topo[:,1] = np.linspace(0,ne-1,num=ne,dtype=int)
topo[:,2] = np.linspace(1,  ne,num=ne,dtype=int)

nod = oti.zeros((nnode_el,ndof))

# Initialize global matrices
Kglob = oti.lil_matrix((4*(ne+1),4*(ne+1)))
Mglob = oti.lil_matrix((4*(ne+1),4*(ne+1)))
Cglob = oti.lil_matrix((4*(ne+1),4*(ne+1)))

# Assembly of global stiffness and mass matrices
K_elems, M_elems, C_elems, DME_list = [], [], [], []
for i in range(ne):
    ind=topo[i,:]
    for k in range(nnode_el):
        nod[k,:] = nodes[int(ind[k+1]),1:]
    
    K=StiffnessMatrix(nod,Emod,bs,ht)
    M=MassMatrix(nod,rho,bs,ht)
    C = alpha_r * M + beta_r * K

    Assembly(K,ind,Kglob)
    Assembly(M,ind,Mglob)
    Assembly(alpha_r*M+beta_r*K,ind,Cglob)

# Apply boundary conditions by imposing large diagonal terms (TGV penalty method)
TGV = 1.0e20

Kglob2 = Kglob.copy()
Kglob2[0,0] = Kglob2[0,0]+ TGV
Kglob2[1,1] = Kglob2[1,1]+ TGV
Kglob2[2,2] = Kglob2[2,2]+ TGV
Kglob2[3,3] = Kglob2[3,3]+ TGV


Mglob2 = Mglob.copy()
Mglob2[0,0] = Mglob2[0,0] + TGV
Mglob2[1,1] = Mglob2[1,1] + TGV
Mglob2[2,2] = Mglob2[2,2] + TGV
Mglob2[3,3] = Mglob2[3,3] + TGV

Cglob2 = Cglob.copy()
Cglob2[0,0] = Cglob2[0,0] + TGV
Cglob2[1,1] = Cglob2[1,1] + TGV
Cglob2[2,2] = Cglob2[2,2] + TGV
Cglob2[3,3] = Cglob2[3,3] + TGV

# Convert to CSR format for sparse solver
Kglob2 = Kglob2.tocsr()
Mglob2 = Mglob2.tocsr()
Cglob2 = Cglob2.tocsr()

if calc_eig:
    Kglob3 = np.zeros((2*4*(ne+1),2*4*(ne+1)))
    Mglob3 = np.zeros((2*4*(ne+1),2*4*(ne+1)))

    Kglob3[0:4*(ne+1),0:4*(ne+1)]=Mglob2.real.toarray()
    Kglob3[4*(ne+1):8*(ne+1),4*(ne+1):8*(ne+1)]=Kglob2.real.toarray()

    Mglob3[0:4*(ne+1),4*(ne+1):8*(ne+1)]=-Mglob2.real.toarray()
    Mglob3[4*(ne+1):8*(ne+1),0:4*(ne+1)]=-Mglob2.real.toarray()
    Mglob3[4*(ne+1):8*(ne+1),4*(ne+1):8*(ne+1)]=-Cglob2.real.toarray()
    Mglob3 = Mglob3 + spr.eye(Mglob3.shape[0]) * np.finfo(float).eps

    Kglob2csr = spr.csr_matrix(Kglob3.real)
    Mglob2csr = spr.csr_matrix(Mglob3.real) 
    
    neigs = 40
    eigval,eigvec = spla.eigs(Kglob2csr,neigs,Mglob2csr,which='SM',maxiter=10000)
else:
    # Load external eigenvalues and eigenvectors
    mat_contents=loadmat('data/Nominal_Eigenvalues_Underdamped.mat')
    eigvec=mat_contents['V']
    eigval=mat_contents['W']
    eigval=eigval[:,0]

print('Finished Step #1')

# Compute derivatives for first eigenpair
neig=4
# Number of Variables of interest
Na = 7

# Initialize Arrays to Store OTI version of eigenvalues and eigenvectors (Real and Imaginary)
Phi_res_re=oti.zeros((4 * (ne + 1), neig))
lambda_res_re=oti.zeros((1, neig))

Phi_res_im=oti.zeros((4 * (ne + 1), neig))
lambda_res_im=oti.zeros((1, neig))

print('Began Differentiation Loop')
for i in range(neig):


    lamda_re = eigval[i].real
    lamda_imag = eigval[i].imag
    
    Phi_re   = oti.array(eigvec[:,i].real)
    Phi_imag   = oti.array(eigvec[:,i].imag)

    print(f'Started Step #2 for Eigenvalue {i+1}')

    # Tangent Matrix Estimation
    T = TangentMatrix(Kglob2.real.todense(), Mglob2.real.todense(), Cglob2.real.todense(), lamda_re.real, lamda_imag.real, Phi_re.real.flatten(), Phi_imag.real.flatten() )
    

    nn=(Kglob2.shape[0]+1)
    TT=np.zeros((nn*2,nn*2))

    TT[0:nn,0:nn]=T.real
    TT[nn:2*nn,nn:2*nn]=T.real
    TT[0:nn,nn:2*nn]=-T.imag
    TT[nn:2*nn,0:nn]=T.imag
    print(f'Finished Step #2 for Eigenvalue {i+1}')

    # Tangent Matrix Factorization
    c, low = lu_factor(T) ## Cholesky factorization of T

    # Initialize State Vector
    u_re = oti.zeros((Kglob2.shape[0]+1,1))
    u_im = oti.zeros((Kglob2.shape[0]+1,1))
    
    u_re[:Kglob2.shape[0],0]=Phi_re
    u_re[Kglob2.shape[0],0] =lamda_re

    u_im[:Kglob2.shape[0],0]=Phi_imag
    u_im[Kglob2.shape[0],0] =lamda_imag

    # Loop for Derivatives Computation

    # Definiton of Perturbations to Input Parameters
    print(f'Started Step #3 for Eigenvalue {i+1}')
    Emod=68.9e9+oti.e(1,order=1)
    rho=2770.0+oti.e(2,order=1)
    bs=0.01+oti.e(3,order=1)
    ht=0.015+oti.e(4,order=1)

    alpha_r  = 1.0E-1+oti.e(6,order=1)
    beta_r   = 1.0E-2+oti.e(7,order=1)

    nodes = oti.zeros((ne+1,ndof+1))
    nodes[:,0] = oti.array(np.linspace(0, ne, num=ne+1))

    nodes[:,1] = oti.array(np.linspace(0.0, L,num=ne+1))
    nodes[:,1] = nodes[:,1]+nodes[:,1]*oti.e(5,order=1)/L
    
    for ordi in range(1,maxOrder+1):
        # Define Truncation Order
        Emod += 0*oti.e(1,order=ordi)
        rho += 0*oti.e(1,order=ordi)
        bs += 0*oti.e(1,order=ordi)
        ht += 0*oti.e(1,order=ordi)
        alpha_r  += 0*oti.e(1,order=ordi)
        beta_r   += 0*oti.e(1,order=ordi)
        nodes[:,1] += 0*oti.e(1,order=ordi)
        nod = oti.zeros((nnode_el,ndof))

        # Evaluate OTI Residual
        Kglob = oti.lil_matrix((4*(ne+1),4*(ne+1)))
        Mglob = oti.lil_matrix((4*(ne+1),4*(ne+1)))
        Cglob = oti.lil_matrix((4*(ne+1),4*(ne+1)))

        for conti in range(ne):
            ind=topo[conti,:]
            for k in range(nnode_el):
                nod[k,:] = nodes[int(ind[k+1]),1:]
            
            K=StiffnessMatrix(nod,Emod,bs,ht)
            M=MassMatrix(nod,rho,bs,ht)
            Assembly(K,ind,Kglob)
            Assembly(M,ind,Mglob)
            Assembly(alpha_r*M+beta_r*K,ind,Cglob)

        Kglob2 = Kglob.copy()
        Kglob2[0,0] =TGV
        Kglob2[1,1] =TGV
        Kglob2[2,2] =TGV
        Kglob2[3,3] =TGV


        Mglob2 = Mglob.copy()
        Mglob2[0,0] = TGV
        Mglob2[1,1] = TGV
        Mglob2[2,2] = TGV
        Mglob2[3,3] = TGV

        Cglob2 = Cglob.copy()
        Cglob2[0,0] = TGV
        Cglob2[1,1] = TGV
        Cglob2[2,2] = TGV
        Cglob2[3,3] = TGV

        Kglob2 = Kglob2.tocsr()
        Mglob2 = Mglob2.tocsr()
        Cglob2 = Cglob2.tocsr()


        Phi_re = u_re[:Kglob2.shape[0],0] + 0*oti.e(1,order=ordi)
        lamda_re = u_re[Kglob2.shape[0],0] + 0*oti.e(1,order=ordi)

        Phi_imag = u_im[:Kglob2.shape[0],0] + 0*oti.e(1,order=ordi)
        lamda_imag = u_im[Kglob2.shape[0],0] + 0*oti.e(1,order=ordi)

        u_re[Kglob2.shape[0],0] =lamda_re
        u_im[Kglob2.shape[0],0] =lamda_imag

        res_re, res_im = global_residual(Kglob2, Mglob2, Cglob2, lamda_re, Phi_re, lamda_imag, Phi_imag)
  
        # Extraction of OTI coefficients to form Matrix R
        rhsi_re = oti.get_order_im_array(ordi,res_re)
        rhsi_im = oti.get_order_im_array(ordi,res_im)
        rhsi=rhsi_re+rhsi_im*1j
        
        # Solution of Linear Systems
        u_i = lu_solve((c, low), -rhsi)
        # Update of state vector with derivatives
        oti.set_order_im_from_array(ordi,u_i.real,u_re)
        oti.set_order_im_from_array(ordi,u_i.imag,u_im)

    
    
    Phi_res_re[:,i]  = u_re[:Kglob2.shape[0],0]
    lambda_res_re[0,i] = u_re[Kglob2.shape[0],0]

    Phi_res_im[:,i]  = u_im[:Kglob2.shape[0],0]
    lambda_res_im[0,i] = u_im[Kglob2.shape[0],0]
    print(f'Finished Step #3 for Eigenvalue {i+1}')



# Extraction of sensitivities of interest
# ---------------------------------------
print(f'Started Step #4')
# Nominal (0th-order) eigenvalues:
eigvals = lambda_res_im.real

# Example: Extract 4th-order eigenvalue derivative w.r.t variable 2 (ρ - mass density)
der1_val = lambda_res_im.get_deriv([[1, 1]])

print(der1_val)
# Example: Extract 4th-order eigenvector derivative w.r.t variable 2 (ρ - mass density)
der1_vec = Phi_res_re.get_deriv([[2, 4]])

# Example: Extract 5th-order eigenvalue mixed derivative w.r.t all 5 design variables
der2_val = lambda_res_im.get_deriv([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]])


# General extraction explanation:
# - The argument to `get_deriv()` is a list of lists:
#   Each sublist: [variable_index, order_of_derivative]
#   Example: [2, 4] means 4th derivative w.r.t design variable 2.
# - Higher-order mixed derivatives follow the same syntax:
#   Example: [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]] for 1st derivatives w.r.t all five variables.

# Loop to collect all derivatives up to maxOrder into an array `der_eigvals`

der_eigvals=np.zeros((neig,coti.ndir_total(Na,maxOrder)))
for i in range(neig):
    print(i)
    der_eigvals_i=lambda_res_im[0,i].real
    for ordi in range(1, maxOrder + 1):
        num = oti.array([lambda_res_im[0,i]])
        
        # Extract coefficients for derivative order `ordi` from OTI representation
        cnum = oti.get_order_im_array(ordi, num)
        # Append this derivative order to the output array
        der_eigvals_i = np.append(der_eigvals_i, cnum)
    der_eigvals[i,:]=der_eigvals_i


# Save all extracted derivatives to MATLAB .mat file for postprocessing
mdic = {"der_eigvals": der_eigvals}
os.makedirs('results', exist_ok=True)
savemat("results/DerivativesEigenvaluesUnderdamped.mat", mdic)






    