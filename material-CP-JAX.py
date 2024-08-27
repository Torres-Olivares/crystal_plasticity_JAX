# --------------------------------------------------------------------------------------
# IMPORT THE REQUIERED LIBRARIES
# --------------------------------------------------------------------------------------
import numpy as np
import jax
import jax.numpy as jnp

import time

from scipy.spatial.transform import Rotation as R

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # use double-precision


# --------------------------------------------------------------------------------------
# DEFINE MATERIAL PROPERTIES
# --------------------------------------------------------------------------------------
mat_prop = {
    "YoungModulus": 75.76e9,
    "PoissonRatio": 0.334, 
    "slip_rate": 0.001,
    "exponent_m": 20,
    "hardening_slope": 1000000000,
    # "hardening_slope": 0.0000001,
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


def mandel_4o_to_matrix(X):
    return True


def mandel_matrix_to_4o(X):
    return True



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

#@jax.jit
def outer_product_2o_2o(T2_a, T2_b):
    """
    Performs the outer product between two second-order tensors.
    T2_a is a tensor with dimensions [i,j] and T2_b is a tensor with dimensions [k,l].
    The result is a fourth-order tensor with dimensions [i,j,k,l].
    """
    result = jnp.einsum('ij,kl->ijkl', T2_a, T2_b)
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



# --------------------------------------------------------------------------------------
# MAIN CP FUNCTIONS
# --------------------------------------------------------------------------------------
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

    # Calculando dCe_dFe sumando los productos tensoriales adecuados
    dCe_dFe = simple_contraction_4o_2o(I4_T, Fe) + simple_contraction_2o_4o(Fe_T, I4)

    # Calculando dS_dFe utilizando la contracción doble
    dS_dFe = double_contraction_4o_4o(D4, 0.5 * dCe_dFe)

    # Calculando dA_dFe usando las multiplicaciones definidas
    dA_dFe = simple_contraction_4o_2o(dS_dFe, Ce) + simple_contraction_2o_4o(S, dCe_dFe)

    # Calculando dFe_dLp_trial
    dFe_dLp_trial = -delta_t * simple_contraction_2o_4o(F @ jnp.linalg.inv(Fp), I4)

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


# #@jax.jit
def material_model(F, Fp_prev, Lp_prev, P0_sn, resistance,del_time):
    '''
    Function to compute all the quantities related to the amterial model

    In:  
    - F: Deformation gradient.
    - Fp_prev: Previous plastic deformation gradient
    - Lp_prev: Plastic velocity gradient (vectorial format)
    - P0_sn: schmidth tensor (sl_0 x nl_0)
    - Resistance: Resistance of all the slip systems
    - del_time: time step

    Out: 
    - S: Second Piola Kirchhoff
    - Fp: Plastic deformation gradient of the current step
    - Lp_trial: Velocity gradient of the current step
    - new_resistance: Updated slip resistances
    '''

    E = mat_prop['YoungModulus']
    nu = mat_prop['PoissonRatio']
    gamma_dot_0 = mat_prop['slip_rate']
    s_inf = mat_prop['saturation_strenght']
    h_0 = mat_prop['hardening_slope']
    m = mat_prop['exponent_m']
    a = mat_prop['exponent_a']
    q = mat_prop['q']

    D4 = fourth_order_elasticity(E,nu)

    # Loop for the optimizer to find the best Lp
    Lp_trial = Lp_prev
    limit_iterations = 1000
    for iteration in range(limit_iterations):

        # Plastic deformation gradient
        Fp = plastic_deformation_gradient(Lp_trial, Fp_prev, del_time)

        # Elastic deformation gradient
        Fe = elastic_deformation_gradient(F, Fp)

        # Right Cauchy-Green strain
        r_cauchy = Fe.T @ Fe

        # Green lagrange strain
        g_lagrange =  0.5*(r_cauchy - jnp.eye(3))

        # Second Piola-Kirchhoff stress
        S = double_contraction_4o_2o(D4, g_lagrange)

        # Calculate resolved shear stress for all slip systems
        tau = resolved_shear(P0_sn, S, r_cauchy)

        # Calculate plastic shear rate for all slip systems in parallel
        gamma_dot = plastic_shear_rate(tau, resistance, gamma_dot_0, m)

        # Calculate plastic velocity gradient for all slip systems in parallel
        Lp_calc = plastic_velocity_gradient(gamma_dot, P0_sn)

        # Calculate the residual
        Residual = Lp_trial - Lp_calc

        # Check for convergence
        if jnp.linalg.norm(Residual) < 1e-9:
            # print(f"CP converged after {iteration+1} iterations !!!!!!!!!!!!")
            break
        
        # NR algorithm jacobian and correction of the trial value
        jacobian_R_L = derivative_R_wrt_Lp_trial(tau,P0_sn,gamma_dot_0,resistance,m,Fe,S,r_cauchy,del_time,Fp_prev,F,D4)
        correction = double_contraction_4o_2o(invert_4o_tensor(jacobian_R_L), Residual)
        Lp_trial = Lp_trial -  correction

    # Actualize for the new resistance after convergence
    s_dot = slip_resistance_rate(gamma_dot, h_0, resistance, s_inf, a, q)
    new_resistance = resistence_integration(resistance, s_dot, del_time)

    return S, Fp, Lp_trial, new_resistance, iteration+1


@jax.jit
def material_model_jit(F, Fp_prev, Lp_prev, P0_sn, resistance, del_time):
    '''
    Function to compute all the quantities related to the material model
    
    In:
    - F: Deformation gradient.
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
        S = double_contraction_4o_2o(D4, g_lagrange)
        
        # Calculate resolved shear stress for all slip systems
        tau = resolved_shear(P0_sn, S, r_cauchy)
        
        # Calculate plastic shear rate for all slip systems in parallel
        gamma_dot = plastic_shear_rate(tau, resistance, gamma_dot_0, m)
        
        # Calculate plastic velocity gradient for all slip systems in parallel
        Lp_calc = plastic_velocity_gradient(gamma_dot, P0_sn)
        
        # Calculate the residual
        Residual = Lp_trial - Lp_calc
        
        # Check for convergence
        converged = jnp.linalg.norm(Residual) < 1e-9
        
        # NR algorithm jacobian and correction of the trial value
        jacobian_R_L = derivative_R_wrt_Lp_trial(tau, P0_sn, gamma_dot_0, resistance, m, Fe, S, r_cauchy, del_time, Fp_prev, F, D4)
        # correction = double_contraction_4o_2o(invert_4o_tensor(jacobian_R_L), Residual)
        correction = jax.scipy.linalg.solve(jacobian_R_L.reshape(9, 9) , Residual.reshape(9,1))
        Lp_trial = Lp_trial - correction.reshape(3,3)
        
        iteration = iteration + 1

        return iteration, converged, S, Fp, Lp_trial, gamma_dot
 
    def cond_fn(state):
        iteration, converged, _, _, _, _ = state
        return jnp.logical_and(iteration < 1000, jnp.logical_not(converged))
    
    # Initialize the state with dummy values
    initial_state = (0, False, jnp.zeros((3, 3)), Fp_prev, Lp_prev, jnp.zeros(len(resistance)))
    
    # Run the while loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    iteration, _, S, Fp, Lp_trial, gamma_dot = final_state
        
    # Actualize for the new resistance after convergence
    s_dot = slip_resistance_rate(gamma_dot, h_0, resistance, s_inf, a, q)
    new_resistance = resistence_integration(resistance, s_dot, del_time)
    
    return S, Fp, Lp_trial, new_resistance, iteration, gamma_dot


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

# angles 12, 23 and 13
initial_angles = [16.0, 14.0, 72.0]

# Define the time-steps
ttime = 200
steps = 200
epsilon_dot_0 = 0.001/ttime

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
eps_t = [] 
Fs = []  # Lista para almacenar los tensores F
for step in range(0, steps ):  # Comenzamos desde 1 hasta steps
    t = step * delta_time 
    lamb = 1+epsilon_dot_0 * t    # lineal
    F = jnp.array([[lamb,       0.0, 0.0], 
                  [0.0, lamb**(-1), 0.0], 
                  [0.0,        0.0, 1.0]])
    Fs.append(F)
    eps_t.append(lamb-1)
    if step == steps:
        break


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
stresst = 0


for F in Fs:    

    if step_count == 2:
        start_time = time.time()

    # Calculate the crystal plasticity tensors
    stress, Fp, Lp, resistance, ite, gam_dot = material_model_jit(F, Fp, Lp, element_P0_sn, resistance, delta_time)

    sigma = 1/np.linalg.det(F)*(np.linalg.inv(F) @ stress @ F.T)

    # Append results for visualization
    sigma_t.append(sigma[(0,0)])
    gamma_dot_t.append(gam_dot)
    strain.append(F[(0,0)]-1)


    step_count += 1


    print('STEP ',step_count,'-------- converged on ',ite, 'iterations --------------------')

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
plt.figure()
plt.plot(strain,sigma_t, "-oC3", label="FEM")
plt.xlabel(r"Strain $\varepsilon_{xx}$")
plt.ylabel(r"Stress $\sigma_{xx}$ [MPa]")
plt.legend()
plt.grid()
plt.show()
# plt.savefig('plot1.png')


plt.figure()
for i in range(len(sl_0)):
    shear_rate = [abs(inner_list[i]) for inner_list in gamma_dot_t]
    plt.plot(strain, shear_rate, label=f'Slip system {i+1}')
plt.xlabel(r"Strain $\varepsilon_{xx}$")
plt.ylabel(r"Stress $\sigma_{xx}$ [MPa]")
plt.legend()
plt.grid()
plt.show()
# plt.savefig('plot1.png')