# -----------------------------------------------------------------------------------
# IMPORT THE REQUIERED LIBRARIES
# -----------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from mpi4py import MPI
import ufl
import basix
from dolfinx import fem, io
from dolfinx.common import Timer

from petsc4py import PETSc
from dolfinx import fem
from dolfinx.mesh import create_box, CellType
import dolfinx.fem.petsc

import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R
from jax.scipy.linalg import eigh


import time

jax.config.update("jax_enable_x64", True)  # use double-precision



# -----------------------------------------------------------------------------------
# DEFINE MATERIAL PROPERTIES
# -----------------------------------------------------------------------------------
mat_prop = {
    "YoungModulus": 75.76e9,
    "PoissonRatio": 0.334, 
    "slip_rate": 0.001,
    "exponent_m": 20,
    "hardening_slope": 1000000000,
    "yield_resistance": 2.69e+6,    #tau_0 - s_0
    "saturation_strenght": 67.5e+6,
    "exponent_a": 5.4,
    "q": 1.4
    }



# --------------------------------------------------------------------------------------
# TENSOR-VECTOR TRANSFORMATIONS (MANDELL)
# --------------------------------------------------------------------------------------
def mandel_vector_to_2o(X):
    return jnp.array([[               X[0], np.sqrt(1/2) * X[5], np.sqrt(1/2) * X[4]],
                      [np.sqrt(1/2) * X[5],                X[1], np.sqrt(1/2) * X[3]],
                      [np.sqrt(1/2) * X[4], np.sqrt(1/2) * X[3],                X[2]]])


def mandel_2o_to_vector(X):
    return jnp.array([X[0, 0], 
                      X[1, 1], 
                      X[2, 2], 
                      np.sqrt(2) * X[1, 2], 
                      np.sqrt(2) * X[0, 2], 
                      np.sqrt(2) * X[0, 1]])


def mandel_4o_to_matrix(C):
    # Initialize the Mandel 6x6 matrix
    C_mandel = jnp.zeros((6, 6))

    # Map from (i,j) to Mandel indices (I,J)
    index_map = {
        (0, 0): 0, (1, 1): 1, (2, 2): 2,
        (1, 2): 3, (2, 1): 3,
        (0, 2): 4, (2, 0): 4,
        (0, 1): 5, (1, 0): 5
    }
    
    sqrt2 = np.sqrt(2)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    I = index_map[(i, j)]
                    J = index_map[(k, l)]

                    # Scaling factor for shear components
                    scale = 1.0
                    if (i != j): scale *= sqrt2
                    if (k != l): scale *= sqrt2

                    C_mandel = C_mandel.at[I, J].add(C[i, j, k, l] * scale)

    return C_mandel


def mandel_matrix_to_4o(X):
    return True


def as_3x3_tensor(X):
    return jnp.array([[X[0], X[1], X[2]], 
                      [X[3], X[4], X[5]], 
                      [X[6], X[7], X[8]]])




# --------------------------------------------------------------------------------------
# BASIC TENSOR OPERATIONS
# double_contraction_4o_2o means double contraction of the 4th order N and 2nd order M (N:M)
# --------------------------------------------------------------------------------------
#@jax.jit
def double_contraction_4o_2o(T4, T2):
    """
    Performs the double inner product between a fourth-order and second-order tensors.
    T4 is a tensor with dimensions [i,j,k,l] and T2 is a tensor with dimensions [k,l].
    The result is a second-order tensor with dimensions [i,j].
    """
    result = jnp.einsum('ijkl,kl->ij', T4, T2)
    return result


#@jax.jit
def double_contraction_2o_4o(T2, T4):
    """
    Performs the double inner product between a fourth-order and second-order tensors.
    T4 is a tensor with dimensions [i,j,k,l] and T2 is a tensor with dimensions [i,j].
    The result is a second-order tensor with dimensions [k,l].
    """
    result = jnp.einsum('ij,ijkl->kl', T2, T4)
    return result


#@jax.jit
def double_contraction_4o_4o(T4_a, T4_b):
    """
    Performs the double inner product between two fourth-order tensors.
    T4_a has tensor with dimensions [i,j,k,l] and T4_b has [k,l,m,n].
    The result is a fourth-order tensor with dimensions [i,j,m,n].
    """
    result = jnp.einsum('ijkl,klmn->ijmn', T4_a, T4_b)
    return result


#@jax.jit
def double_contraction_3o_2o(T3, T2):
    """
    Performs the double inner product between a third-order and second-order tensors.
    T3 is a tensor with dimensions [i,m,n] and T2 is a tensor with dimensions [m,n].
    The result is a vector with dimensions [i].
    """
    result = jnp.einsum('imn,mn->i', T3, T2)
    return result


#@jax.jit
def simple_contraction_2o_4o(T2, T4):
    """
    Performs the simple inner product between a second-order and fourth-order tensors.
    T2 is a tensor with dimensions [i,m] and T4 is a tensor with dimensions [m,j,k,l].
    The result is a fourth-order tensor with dimensions [i,j,k,l].
    """
    result = jnp.einsum('im,mjkl->ijkl', T2, T4)
    return result


#@jax.jit
def simple_contraction_4o_2o(T4, T2):
    """
    Performs the simple inner product between a fourth-order and second-order tensors.
    T4 is a tensor with dimensions [i,j,k,m] and T2 is a tensor with dimensions [m,l].
    The result is a fourth-order tensor with dimensions [i,j,k,l].
    """
    result = jnp.einsum('ijkm,ml->ijkl', T4, T2)
    return result


# @jax.jit
def simple_contraction_2o_2o(T2_a, T2_b):
    """
    Performs the simple inner product between a fourth-order and second-order tensors.
    T4 is a tensor with dimensions [i,j,k,m] and T2 is a tensor with dimensions [m,l].
    The result is a fourth-order tensor with dimensions [i,j,k,l].
    """
    result = jnp.einsum('ik,kl->il', T2_a, T2_b)
    return result


#@jax.jit
def outer_product_2o_2o(T2_a, T2_b):
    """
    Performs the outer product between two second-order tensors.
    T2_a is a tensor with dimensions [i,j] and T2_b is a tensor with dimensions [k,l].
    The result is a fourth-order tensor with dimensions [i,j,k,l].
    """
    result = jnp.einsum('ij,kl->ijkl', T2_a, T2_b)
    return result


def tensor_product_2o_2o(T2_a, T2_b):
    """
    Performs the outer product between two second-order tensors.
    T2_a is a tensor with dimensions [i,j] and T2_b is a tensor with dimensions [k,l].
    The result is a fourth-order tensor with dimensions [i,j,k,l].
    """
    result = jnp.einsum('ij,kl->iljk', T2_a, T2_b)
    return result


#@jax.jit
def fourth_order_identity():
  """Creates a fourth-order identity tensor (d_ik d_jl  e_i x e_j x e_k x e_l)."""
  I = jnp.eye(3)
  I4 = jnp.einsum('ik,jl->ijkl', I, I)
  return I4


#@jax.jit
def fourth_order_identity_transpose():
  """Creates a fourth-order identity tensor (d_il d_jk  e_i x e_j x e_k x e_l)."""
  I = jnp.eye(3)
  I4 = jnp.einsum('il,jk->ijkl', I, I)
  return I4


#@jax.jit
def invert_4o_tensor(array):
  """Inverts a 3x3x3x3 array by reshaping to a 9x9 matrix, inverting, and reshaping back.

  Args:
    array: The 3x3x3x3 array to invert.

  Returns:
    The inverted 3x3x3x3 array.
  """
  reshaped_array = array.reshape(9, 9)
  inverted_array = jnp.linalg.inv(reshaped_array)
  return inverted_array.reshape(3, 3, 3, 3)


#@jax.jit
def fourth_order_elasticity(E, nu):
  """Calculates the 4th order elasticity tensor"""
  
  I4 = fourth_order_identity()
  I4_t = fourth_order_identity_transpose()
  I = jnp.eye(3)

  lmbda = E*nu/((1+nu)*(1-2*nu))
  mu = E/(2*(1+nu))

  D4 = lmbda*(outer_product_2o_2o(I,I)) + mu*(I4 + I4_t)
  
  return D4


# -----------------------------------------------------------------------------------
# MATERIAL CONSTITUTIVE BEHAVIOR (JAX-SIDE)
# -----------------------------------------------------------------------------------
#@jax.jit
def slip_systems_fcc():
    '''
    DEFINE THE VECTORS S AND N FOR THE SLIP SYSTEMS
    fcc crystal case (12 slip systems)
    '''
    # Slip systems for a fcc crystal
    sl_0 = jnp.array([[ 0.0, 1.0,-1.0],
                    [-1.0, 0.0, 1.0],
                    [ 1.0,-1.0, 0.0],
                    [ 0.0,-1.0,-1.0],
                    [ 1.0, 0.0, 1.0],
                    [-1.0, 1.0, 0.0],
                    [ 0.0,-1.0, 1.0],
                    [-1.0, 0.0,-1.0],
                    [ 1.0, 1.0, 0.0],
                    [ 0.0, 1.0, 1.0],
                    [ 1.0, 0.0,-1.0],
                    [-1.0,-1.0, 0.0]])

    nl_0 = jnp.array([[ 1.0, 1.0, 1.0],
                    [ 1.0, 1.0, 1.0],
                    [ 1.0, 1.0, 1.0],
                    [-1.0,-1.0, 1.0],
                    [-1.0,-1.0, 1.0],
                    [-1.0,-1.0, 1.0],
                    [ 1.0,-1.0,-1.0],
                    [ 1.0,-1.0,-1.0],
                    [ 1.0,-1.0,-1.0],
                    [-1.0, 1.0,-1.0],
                    [-1.0, 1.0,-1.0],
                    [-1.0, 1.0,-1.0]])

    # Normalize each slip system
    sl_0 = sl_0 / jnp.linalg.norm(sl_0, axis=1)[:, jnp.newaxis]
    nl_0 = nl_0 / jnp.linalg.norm(nl_0, axis=1)[:, jnp.newaxis]

    return sl_0, nl_0


#@jax.jit
def slip_systems_1slip():
    '''
    DEFINE THE VECTORS S AND N FOR THE SLIP SYSTEMS
    Example case of only 1 slip system
    '''
    # Example of one slip system 
    sl_0 = jnp.array([[ 0.0, 1.0,-1.0]])
    nl_0 = jnp.array([[ 1.0, 1.0, 1.0]])

    # Normalize each slip system
    sl_0 = sl_0 / jnp.linalg.norm(sl_0, axis=1)[:, jnp.newaxis]
    nl_0 = nl_0 / jnp.linalg.norm(nl_0, axis=1)[:, jnp.newaxis]

    return sl_0, nl_0


#@jax.jit
def plastic_deformation_gradient(Lp,Fp_old,delta_t):
    inverse_part = (jnp.eye(3) - delta_t * Lp)
    Fp_new = jnp.linalg.solve(inverse_part, Fp_old)
    return Fp_new


#@jax.jit
def elastic_deformation_gradient(F,Fp):
    Fe = F @ jnp.linalg.inv(Fp)
    return Fe


#@jax.jit
def resistence_integration(s_alpha, s_alpha_dot,delta_t):
    s_alpha_new = s_alpha + s_alpha_dot*delta_t
    return s_alpha_new


#@jax.jit
def slip_resistance_rate(gamma_dot,h_0, s_global, s_inf, a, q):
    '''
    Calculate the resistance rate of all the slip systems
    
    In:
     - gamma_dot: Shear slip rate (vector)
     - h_0: initial hardening modulus (constant)
     - s_global: Resistance of all the slip systems (vector)
     - s_inf: Saturation resistance value (constant)
     - a: Calibration parameter (constant)
     - q: Latent hardening ratio (constant)
    
    Out: 
     - s_dot_alpha: Resistance rate of all the slip systems (vector)
    '''
    num_slips = s_global.shape[0]
    
    # Create a matrix of (1 - s_global / s_inf)**a
    s_ratio = (1 - s_global / s_inf)**a
    s_ratio_matrix = jnp.outer(s_ratio, jnp.ones(num_slips))
    
    # Create the base matrix
    h_matrix = h_0 * s_ratio_matrix * q
    
    # Adjust the diagonal elements ad compute hardening modulus matrix
    diagonal_mask = jnp.eye(num_slips, dtype=bool)
    h_matrix = jnp.where(diagonal_mask, h_0 * s_ratio, h_matrix)

    # Compute the resistance rates using matrix multiplication
    s_dot_alpha = jnp.dot(h_matrix, jnp.absolute(gamma_dot))
    
    return s_dot_alpha


#@jax.jit
def plastic_shear_rate(tau, s, gamma_dot_0, m):
    '''
    PLASTIC SHEAR RATE

    In:  
    - tau: Resolved shear stress (vector)
    - s: Slip resistance (vector)
    - gamma_dot_0: Reference plastic shear rate (constant)
    - m: Rate sensitivity (constant)

    Out: 
    - gamma_dot: Plastic shear rate (vector)
    '''
    gamma_dot = gamma_dot_0 * (tau / s) * jnp.absolute(tau / s)**(m - 1)
    return gamma_dot


#@jax.jit
def resolved_shear(P0_sn,second_piola,right_cauchy):
    '''
    RESOLVED SHEAR STRESS

    In:  
    - P0_sn: matrix containing all the n Tensor for slip systems (n x 3 x 3)
    - second_piola: Second Piola Kirchhoff stress (3 x 3)
    - right_cauchy: Right Cauchy strain (3 x 3)

    Out: 
    - rShear: Resolved shear stress (n)
    '''

    product_tensor = second_piola @ right_cauchy

    # Perform the double contraction with the product tensor for each slip system
    rShears = double_contraction_3o_2o(P0_sn, product_tensor)

    return rShears


#@jax.jit
def plastic_velocity_gradient(gamma_dot, P0_sn):
    '''
    PLASTIC VELOCITY GRADIENT

    In:  
    - gamma_dot: Array of plastic shear rates [12].
    - P0_sn: Matrix of reference slip directions and normals to slip planes [12, 3, 3].

    Out: 
    - Lp: Plastic velocity gradient [3, 3].
    '''
    Lp  = jnp.einsum('i,ijk->jk', gamma_dot, P0_sn)

    return Lp


#@jax.jit
def derivative_Lp_wrt_A(tau, P0_sn, gamma_dot_0, resistance, m):
    """
    Computes the derivative of the velocity gradient with respect to A = (S C)
    
    Parameters:
    - tau: Resolved shear (vector)
    - P0_sn: All the 3x3 slip systems (nx3x3 tensor)
    - gamma_dot_0: Initial slip rate (constant)
    - resistance: Shear resistances (vector)
    - m: Exponent for the phenomenological power law (constant)
    
    Returns:
    - dLp_dA: Derivative of the velocity gradient (4th order tensor)
    """
    dgamma_dtau = gamma_dot_0 * (m / resistance) * (jnp.abs(tau) / resistance) ** (m - 1)
    
    # Compute the tensor product P0_sn[b, i, j] * P0_sn[b, k, l]
    tensor_fourth_order = jnp.einsum('bij,bkl->bijkl', P0_sn, P0_sn)
    
    # Compute dLp_dA using einsum
    dLp_dA = jnp.einsum('b,bijkl->ijkl', dgamma_dtau, tensor_fourth_order)
    
    return dLp_dA


#@jax.jit
def derivative_A_wrt_Lp_trial(Fe, S, Ce, delta_t, Fp, F, D4):
    '''
    Computes the derivative of A = (S C) with respect to the trial velocity gradient

    In:  
    - Fe: Elastic deformation gradient of the current time step Fe(i)
    - S: Second Piola Kirchhoff stress tensor
    - Ce: Right Cauchy-Green strain tensor
    - delta_t: Well... delta time...
    - Fp: Plastic deformation gradient of the previous time step Fp(i-1)
    - F: Deformation gradient of the current time step F(i)
    - D4: 4th order elasticity tensor

    Out: 
    - dA_dLp_trial: Derivative of A = (S C) with respect to the trial velocity gradient
    '''
    Fe_T = Fe.T
    I4 = fourth_order_identity()
    I4_T = fourth_order_identity_transpose()
    I2 = jnp.eye(3)

    # Calculando dCe_dFe sumando los productos tensoriales adecuados
    # dCe_dFe = simple_contraction_4o_2o(I4_T, Fe) + simple_contraction_2o_4o(Fe_T, I4)
    dCe_dFe = double_contraction_4o_4o(tensor_product_2o_2o(I2,Fe), I4_T) + tensor_product_2o_2o(Fe_T, I2)

    # Calculando dS_dFe utilizando la contracción doble
    dSe_dFe = double_contraction_4o_4o(D4, 0.5 * dCe_dFe)

    # Calculando dA_dFe usando las multiplicaciones definidas
    dA_dFe = double_contraction_4o_4o(tensor_product_2o_2o(I2,Ce),dSe_dFe) + double_contraction_4o_4o(tensor_product_2o_2o(S,I2),dCe_dFe)

    # Calculando dFe_dLp_trial
    dFe_dLp_trial = -delta_t * double_contraction_4o_4o(tensor_product_2o_2o(F,I2), tensor_product_2o_2o(jnp.linalg.inv(Fp),I2))

    # Perform double inner product
    dA_dLp_trial = double_contraction_4o_4o(dA_dFe, dFe_dLp_trial)

    return dA_dLp_trial


#@jax.jit
def derivative_R_wrt_Lp_trial(tau, P0_sn, gamma_dot_0, resistance, m, Fe, S, Ce, delta_t, Fp, F, D4):
    """
    Computes the derivative of the velocity gradient with respect to A = (S C)
    
    Parameters:
    - tau: Resolved shear (vector)
    - P0_sn: All the 3x3 slip systems (nx3x3 tensor)
    - gamma_dot_0: Initial slip rate (constant)
    - resistance: Shear resistances (vector)
    - m: Exponent for the phenomenological power law (constant)
    - Fe: Elastic deformation gradient of the current time step Fe(i)
    - S: Second Piola Kirchhoff stress tensor
    - Ce: Right Cauchy-Green strain tensor
    - delta_t: Well... delta time...
    - Fp: Plastic deformation gradient of the previous time step Fp(i-1)
    - F: Deformation gradient of the current time step F(i)
    - D4: 4th order elasticity tensor
    
    Returns:
    - dR_dLp_trial: Derivative of the residual with respect to the trial velocity gradient
    """
    dLp_dA = derivative_Lp_wrt_A(tau, P0_sn, gamma_dot_0, resistance, m)

    dA_dLp_trial = derivative_A_wrt_Lp_trial(Fe, S, Ce, delta_t, Fp, F, D4)

    dLp_dLp_trial = double_contraction_4o_4o(dLp_dA, dA_dLp_trial)

    dR_dLp_trial = fourth_order_identity() - dLp_dLp_trial

    return dR_dLp_trial


def compute_U(Cc):
    # Compute eigenvalues and eigenvectors of C
    eigvals, eigvecs = eigh(Cc)

    # Compute principal strains
    principal_strains = jnp.sqrt(eigvals)

    # Compute U (right stretch tensor)
    U = jnp.einsum('i,ji,ki->jk', principal_strains, eigvecs, eigvecs)
    return U


# Compute the Jacobian (derivative) of U with respect to C
compute_dU_dC = jax.jacobian(compute_U)

# Function to check if a matrix is NaN and replace if necessary
def check_and_replace_with_elastic(K_mat,D4):
    # Compute the norm and check if it's NaN
    norm_is_nan = jnp.isnan(jnp.linalg.norm(K_mat))
    
    # Use jax.lax.cond to select the appropriate tensor
    K_mat_valid = jax.lax.cond(norm_is_nan,
                               lambda: D4,  # If norm is NaN, return elastic tensor
                               lambda: K_mat)  # Else, return the original K_mat
    
    return K_mat_valid


def material_tang(F, Fp, Se, del_t, Fp0, P0_sn, resistance, D4, gamma_dot_0, m):
 

    # Identity tensors to use
    I4 = fourth_order_identity()
    I4_t = fourth_order_identity_transpose()
    I2 = jnp.eye(3)

    #SImple tensor calculations
    Fp_inv = jnp.linalg.inv(Fp)
    Fe = elastic_deformation_gradient(F, Fp)
    Fe_t = Fe.T
    Ce = Fe_t @ Fe
    C = F.T @ F
    Fp0_inv = jnp.linalg.inv(Fp0)

    U = compute_U(C)
    R = F @ jnp.linalg.inv(U)

    # Calculate resolved shear stress for all slip systems
    tau = resolved_shear(P0_sn, Se, Ce)
    dLp_dA = derivative_Lp_wrt_A(tau, P0_sn, gamma_dot_0, resistance, m)


    dF_dU = simple_contraction_2o_4o(R,I4)

    dU_dC = compute_dU_dC(C)

    dC_dE = 2 * I4

    dF_dE = double_contraction_4o_4o(double_contraction_4o_4o(dF_dU,dU_dC),dC_dE)

    dCe_dFe = double_contraction_4o_4o(tensor_product_2o_2o(I2,Fe), I4_t) + tensor_product_2o_2o(Fe_t, I2)

    dSe_dFe = double_contraction_4o_4o(D4, 0.5 * dCe_dFe)

    dA_dSe = simple_contraction_4o_2o(I4, Ce)

    dLp_dSe = double_contraction_4o_4o(dLp_dA, dA_dSe)

    # dSe_dF has 3 sub_parts to make it readable
    dSe_dF_part1 = tensor_product_2o_2o(simple_contraction_2o_2o(F,Fp0_inv),I2)
    dSe_dF_part2 = tensor_product_2o_2o(I2,Fp_inv)
    dSe_dF_inv_part = (I4 + del_t * double_contraction_4o_4o(double_contraction_4o_4o(dSe_dFe,dSe_dF_part1),dLp_dSe))
    # Solve the equation [I + dT dSe_dFe : (F F_p0^{-1} x I) : dLp_dSe] : dSe_dF = dSe_dFe : (I x F_p^{-1})
    dSe_dF = double_contraction_4o_4o(invert_4o_tensor(dSe_dF_inv_part),double_contraction_4o_4o(dSe_dFe,dSe_dF_part2))

    dFp_inv_dF = -del_t * double_contraction_4o_4o(double_contraction_4o_4o(tensor_product_2o_2o(Fp0_inv,I2),dLp_dSe),dSe_dF)

    # dS_dF has 3 sub_parts to make it readable
    dS_dF_part1 = tensor_product_2o_2o(Fp_inv,Fp_inv)
    dS_dF_part2 = tensor_product_2o_2o(I2, simple_contraction_2o_2o(Se,Fp_inv.T))
    dS_dF_part3 = tensor_product_2o_2o(simple_contraction_2o_2o(Fp_inv,Se),I2)
    dS_dF = double_contraction_4o_4o(dS_dF_part1,dSe_dF) + double_contraction_4o_4o(dS_dF_part2,dFp_inv_dF) + double_contraction_4o_4o(double_contraction_4o_4o(dS_dF_part3,I4_t),dFp_inv_dF)

    dS_dE = double_contraction_4o_4o(dS_dF, dF_dE)

    K_mat_checked = check_and_replace_with_elastic(dS_dE,D4)

    return K_mat_checked



@jax.jit
def material_model_jit(def_grad, Fp_prev, Lp_prev, P0_sn, resistance, del_time):
    '''
    Function to compute all the quantities related to the material model
    
    In:
    - def_grad: Deformation gradient.
    - Fp_prev: Previous plastic deformation gradient
    - Lp_prev: Plastic velocity gradient (vectorial format)
    - P0_sn: schmidt tensor (sl_0 x nl_0)
    - Resistance: Resistance of all the slip systems
    - del_time: time step
    
    Out:
    - S: Second Piola Kirchhoff
    - Fp: Plastic deformation gradient of the current step
    - Lp_trial: Velocity gradient of the current step
    - new_resistance: Updated slip resistances
    '''

    F = as_3x3_tensor(def_grad)

    E = mat_prop['YoungModulus']
    nu = mat_prop['PoissonRatio']
    gamma_dot_0 = mat_prop['slip_rate']
    s_inf = mat_prop['saturation_strenght']
    h_0 = mat_prop['hardening_slope']
    m = mat_prop['exponent_m']
    a = mat_prop['exponent_a']
    q = mat_prop['q']
    
    D4 = fourth_order_elasticity(E, nu)
    
    def body_fn(state):
        iteration, converged, _, _, Lp_trial, _ = state

        # Plastic deformation gradient
        Fp = plastic_deformation_gradient(Lp_trial, Fp_prev, del_time)
        
        # Elastic deformation gradient
        Fe = elastic_deformation_gradient(F, Fp)
        
        # Right Cauchy-Green strain
        r_cauchy = Fe.T @ Fe
        
        # Green lagrange strain
        g_lagrange = 0.5 * (r_cauchy - jnp.eye(3))
        
        # Second Piola-Kirchhoff stress
        Se = double_contraction_4o_2o(D4, g_lagrange)
        
        # Calculate resolved shear stress for all slip systems
        tau = resolved_shear(P0_sn, Se, r_cauchy)
        
        # Calculate plastic shear rate for all slip systems in parallel
        gamma_dot = plastic_shear_rate(tau, resistance, gamma_dot_0, m)
        
        # Calculate plastic velocity gradient for all slip systems in parallel
        Lp_calc = plastic_velocity_gradient(gamma_dot, P0_sn)
        
        # Calculate the residual
        Residual = Lp_trial - Lp_calc
        
        # Check for convergence
        converged = jnp.linalg.norm(Residual) < 1e-9
        
        # NR algorithm jacobian and correction of the trial value
        jacobian_R_L = derivative_R_wrt_Lp_trial(tau, P0_sn, gamma_dot_0, resistance, m, Fe, Se, r_cauchy, del_time, Fp_prev, F, D4)
        correction = double_contraction_4o_2o(invert_4o_tensor(jacobian_R_L), Residual)
        Lp_trial = Lp_trial - correction
        
        iteration = iteration + 1

        return iteration, converged, Se, Fp, Lp_trial, gamma_dot
 
    def cond_fn(state):
        iteration, converged, _, _, _, _ = state
        return jnp.logical_and(iteration < 1000, jnp.logical_not(converged))
    
    # Initialize the state with dummy values
    initial_state = (0, False, jnp.zeros((3, 3)), Fp_prev, Lp_prev, jnp.zeros(len(resistance)))
    
    # Run the while loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    iteration, _, Se, Fp, Lp_trial, gamma_dot = final_state

    # Calculate the tangent stiffness
    K_mat = material_tang(F, Fp, Se, del_time, Fp_prev, P0_sn, resistance,D4,gamma_dot_0,m)

    # Calculate the non-elastic 2-PK
    S2pk = jnp.linalg.inv(Fp) @ Se @ jnp.linalg.inv(Fp.T)


    # Actualize for the new resistance after convergence
    s_dot = slip_resistance_rate(gamma_dot, h_0, resistance, s_inf, a, q)
    new_resistance = resistence_integration(resistance, s_dot, del_time)
    
    # Return stress and tangent stiffness in Mandel format
    stress = mandel_2o_to_vector(S2pk)
    tangent = mandel_4o_to_matrix(K_mat)

    return stress, Fp, Lp_trial, new_resistance, tangent



# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# angles 12, 23 and 13
initial_angles = [16.0, 14.0, 72.0]

# Define the time-steps
ttime = 200
steps = 200
epsilon_dot_0 = 0.001/ttime # 5e-6

s_0 = mat_prop['yield_resistance']

# Initialize time
delta_time = ttime/steps

# Initial s and n vectors
sl_0, nl_0 = slip_systems_fcc()

# From Euler angles
rot_euler = R.from_euler('ZXZ', initial_angles, degrees=True)
matrix = rot_euler.as_matrix()

new_sl0 = sl_0 @ matrix.T
new_nl0 = nl_0 @ matrix.T

element_P0_sn = jnp.einsum('bi,bj->bij', new_sl0, new_nl0)


# --------------------------------------------------------------------------------------
# COMPUTE A LIST OF DEFORMATION GRADIENTS
# --------------------------------------------------------------------------------------
# eps_t = [] 
# Fs = []  # Lista para almacenar los tensores F
# for step in range(0, steps ):  # Comenzamos desde 1 hasta steps
#     t = step * delta_time 
#     lamb = 1.0+epsilon_dot_0 * t    # lineal
#     F = jnp.array([[np.sqrt(lamb)**(-1),                 0.0,  0.0], 
#                   [                 0.0, np.sqrt(lamb)**(-1),  0.0], 
#                   [                 0.0,                 0.0, lamb]])
#     Fs.append(F)
#     eps_t.append(lamb-1)
#     if step == steps:
#         break
# Load the data from the text file
Fs = np.loadtxt('deformation_gradients.txt', delimiter=',')

# --------------------------------------------------------------------------------------
# MAIN CODE
# --------------------------------------------------------------------------------------
# Initialize arrys for storage
sigma_t = []
gamma_dot_t = []
strain = []
# Initial Lp
Lp = jnp.zeros([3, 3])
# Initial plastic deformation gradient
Fp = jnp.eye(3)
step_count = 0
# Initiaize resistances
resistance = jnp.ones(len(sl_0))*s_0



for def_grad in Fs:    

    if step_count == 2:
        start_time = time.time()

    # Calculate the crystal plasticity tensors
    Se, Fp, Lp, resistance, Ktan = material_model_jit(def_grad, Fp, Lp, element_P0_sn, resistance, delta_time)

    # Append results for visualization
    sigma_t.append(Se[2])
    # gamma_dot_t.append(gam_dot)
    strain.append(def_grad[8]-1)

    print("STRAIN = ", def_grad[8]-1)
    print("SIGMA = ", Se[2])

    step_count += 1

    print('STEP ',step_count,'-------- converged on iterations --------------------')

    # print(gamma_dot)
    if step_count == steps:
        break

# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time
print(f"Time spent: {elapsed_time} seconds")
# --------------------------------------------------------------------------------------
# OUTPUT DATA VISUALIZATION
# --------------------------------------------------------------------------------------
# Load the data from the text file
results_rve = np.loadtxt('results.txt', delimiter=',')

plt.figure()
# plt.plot(strain,sigma_t, "-oC3", label="FEM")
plt.plot(strain,sigma_t, label="material")
plt.plot(results_rve[:, 0], results_rve[:, 1], label="RVE")
plt.xlabel(r"Strain $\varepsilon_{zz}$")
plt.ylabel(r"Stress $\sigma_{zz}$ [MPa]")
plt.legend()
plt.grid()
plt.show()
plt.savefig('plot1_mat.png')


# plt.figure()
# for i in range(len(sl_0)):
#     shear_rate = [abs(inner_list[i]) for inner_list in gamma_dot_t]
#     plt.plot(strain, shear_rate, label=f'Slip system {i+1}')
# plt.xlabel(r"Strain $\varepsilon_{zz}$")
# plt.ylabel(r"Stress $\sigma_{zz}$ [MPa]")
# plt.legend()
# plt.grid()
# plt.show()
# plt.savefig('plot2_mat.png')

