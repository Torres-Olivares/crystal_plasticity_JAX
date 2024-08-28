# -----------------------------------------------------------------------------------
# Code made by Sebastian Torres-Olivares (s.a.torres.olivares@tue.nl)
# You can use it, but at least say thanks!
# -----------------------------------------------------------------------------------

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
import dolfinx.fem.petsc

import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R
from jax.scipy.linalg import eigh

from opt_einsum import contract
import time

jax.config.update("jax_enable_x64", True)  # use double-precision

# Record the start time
start_time = time.time()



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
    sq = jnp.sqrt(2)
    C_mandel = jnp.array(
        [[   C[0,0,0,0],    C[0,0,1,1],    C[0,0,2,2], sq*C[0,0,1,2], sq*C[0,0,0,2], sq*C[0,0,0,1]],
         [   C[1,1,0,0],    C[1,1,1,1],    C[1,1,2,2], sq*C[1,1,1,2], sq*C[1,1,0,2], sq*C[1,1,0,1]],
         [   C[2,2,0,0],    C[2,2,1,1],    C[2,2,2,2], sq*C[2,2,1,2], sq*C[2,2,0,2], sq*C[2,2,0,1]],
         [sq*C[1,2,0,0], sq*C[1,2,1,1], sq*C[1,2,2,2],  2*C[1,2,1,2],  2*C[1,2,0,2],  2*C[1,2,0,1]],
         [sq*C[0,2,0,0], sq*C[0,2,1,1], sq*C[0,2,2,2],  2*C[0,2,1,2],  2*C[0,2,0,2],  2*C[0,2,0,1]],
         [sq*C[0,1,0,0], sq*C[0,1,1,1], sq*C[0,1,2,2],  2*C[0,1,1,2],  2*C[0,1,0,2],  2*C[0,1,0,1]]]
        )
    return C_mandel


def mandel_matrix_to_4o(X):
    return True


def as_3x3_tensor(X):
    return jnp.array([[X[0], X[1], X[2]], 
                      [X[3], X[4], X[5]], 
                      [X[6], X[7], X[8]]])



# --------------------------------------------------------------------------------------
# BASIC TENSOR OPERATIONS
# Double_contraction_4o_2o means double contraction of the 4th order N and 2nd order M (N:M)
# Identities are created as global constants inmediatly to reduce computation time
# --------------------------------------------------------------------------------------
def second_order_identity():
  """Creates a second-order identity tensor"""
  return jnp.eye(3)
I2 = second_order_identity()


def fourth_order_identity():
  """Creates a fourth-order identity tensor (d_ik d_jl  e_i x e_j x e_k x e_l)."""
  I = jnp.eye(3)
  return jnp.einsum('ik,jl->ijkl', I, I)
I4 = fourth_order_identity()


def fourth_order_identity_transpose():
  """Creates a fourth-order identity tensor (d_il d_jk  e_i x e_j x e_k x e_l)."""
  I = jnp.eye(3)
  return jnp.einsum('il,jk->ijkl', I, I)
I4_t = fourth_order_identity_transpose()



def double_contraction_4o_2o(T4, T2):
    """
    Performs the double inner product between a fourth-order and second-order tensors.
    T4 is a tensor with dimensions [i,j,k,l] and T2 is a tensor with dimensions [k,l].
    The result is a second-order tensor with dimensions [i,j].
    """
    result = jnp.einsum('ijkl,kl->ij', T4, T2)
    return result


def double_contraction_2o_4o(T2, T4):
    """
    Performs the double inner product between a fourth-order and second-order tensors.
    T4 is a tensor with dimensions [i,j,k,l] and T2 is a tensor with dimensions [i,j].
    The result is a second-order tensor with dimensions [k,l].
    """
    result = jnp.einsum('ij,ijkl->kl', T2, T4)
    return result


def double_contraction_4o_4o(T4_a, T4_b):
    """
    Performs the double inner product between two fourth-order tensors.
    T4_a has tensor with dimensions [i,j,k,l] and T4_b has [k,l,m,n].
    The result is a fourth-order tensor with dimensions [i,j,m,n].
    """
    result = jnp.einsum('ijkl,klmn->ijmn', T4_a, T4_b)
    return result


def double_contraction_3o_2o(T3, T2):
    """
    Performs the double inner product between a third-order and second-order tensors.
    T3 is a tensor with dimensions [i,m,n] and T2 is a tensor with dimensions [m,n].
    The result is a vector with dimensions [i].
    """
    result = jnp.einsum('imn,mn->i', T3, T2)
    return result


def simple_contraction_2o_4o(T2, T4):
    """
    Performs the simple inner product between a second-order and fourth-order tensors.
    T2 is a tensor with dimensions [i,m] and T4 is a tensor with dimensions [m,j,k,l].
    The result is a fourth-order tensor with dimensions [i,j,k,l].
    """
    result = jnp.einsum('im,mjkl->ijkl', T2, T4)
    return result


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
    Performs the simple inner product between two second-order tensors.
    T2_a is a tensor with dimensions [i,k] and T2_b is a tensor with dimensions [k,l].
    The result is a second-order tensor with dimensions [i,l].
    """
    result = jnp.einsum('ik,kl->il', T2_a, T2_b)
    return result
    # return jnp.matmul(T2_a, T2_b)


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


def fourth_order_elasticity(E, nu):
  """Calculates the 4th order elasticity tensor"""
  lmbda = E*nu/((1+nu)*(1-2*nu))
  mu = E/(2*(1+nu))

  D4 = lmbda*(outer_product_2o_2o(I2,I2)) + mu*(I4 + I4_t)
  
  return D4


# -----------------------------------------------------------------------------------
# FENICSX RELATED FUNCTIONS
# -----------------------------------------------------------------------------------
# Create anumpy array containing the orientations per element
def create_element_to_tag_map(domain, cell_tags):
    element_to_tag = np.zeros(domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32)
    for tag in np.unique(cell_tags.values):
        cells = cell_tags.find(tag)
        element_to_tag[cells] = tag
    return element_to_tag


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


def calculate_schmidt_tensor(initial_angles, sl_0, nl_0):
    '''
    Calculate the schmidt tensors for all the slip systems.
    Takes as input the initial Euler angles, initial slip direction and normal vector.
    '''
    rot_euler = R.from_euler('ZXZ', initial_angles, degrees=True)
    matrix = rot_euler.as_matrix()
    new_sl0 = jnp.matmul(sl_0 , matrix.T)
    new_nl0 = jnp.matmul(nl_0 , matrix.T)
    return jnp.einsum('bi,bj->bij', new_sl0, new_nl0)


def extract_euler_angles_zxz(file_path):
    rodrigues_vectors = []
    reading_orientations = False
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == '$ElsetOrientations':
                reading_orientations = True
                continue
            elif line.strip() == '$EndElsetOrientations':
                break
            
            if reading_orientations:
                parts = line.split()
                if len(parts) == 4 and parts[0].isdigit():
                    vector = [float(x) for x in parts[1:]]
                    rodrigues_vectors.append(vector)
    
    # Convert Rodrigues vectors to Euler angles (ZXZ convention)
    orientations = []
    for vec in rodrigues_vectors:
        # Convert Rodrigues vector to rotation matrix
        r = R.from_rotvec(vec)
        # Get Euler angles in ZXZ convention (in radians)
        euler = r.as_euler('zxz', degrees=True)
        orientations.append(euler.tolist())
    
    return orientations


# -----------------------------------------------------------------------------------
# DEFINE MESH AND FUNCTIONSPACE
# -----------------------------------------------------------------------------------
length, height = 1, 1

# Read the mesh
mesh_file = "neper_files/n50-id1.msh"
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD)


sl_0, nl_0 = slip_systems_fcc()
orientations = extract_euler_angles_zxz(mesh_file)
P0_sn_list = jnp.array([calculate_schmidt_tensor(ori,sl_0,nl_0) for ori in orientations])



dim = domain.topology.dim
print(f"Mesh topology dimension d={dim}.")

degree = 1
shape = (dim,)
V = fem.functionspace(domain, ("P", degree, shape))

# Create tag function on DG0 space for cell-constant properties
DG0 = fem.functionspace(domain, ("DG", 0))


# -----------------------------------------------------------------------------------
# DEFINE MATERIAL PROPERTIES
# -----------------------------------------------------------------------------------
mat_prop = {
    "YoungModulus": 75.76e9,
    "PoissonRatio": 0.334,
    "Elastic_tangent": fourth_order_elasticity(75.76e9, 0.334), 
    "slip_rate": 0.001,
    "exponent_m": 20,
    "hardening_slope": 1000000000,
    "yield_resistance": 2.69e+6,    #tau_0 - s_0
    "saturation_strenght": 67.5e+6,
    "exponent_a": 5.4,
    "q": 1.4
    }


# -----------------------------------------------------------------------------------
# DEFINE BOUNDARY CONDITIONS
# -----------------------------------------------------------------------------------
V_ux, _ = V.sub(0).collapse()
V_uy, _ = V.sub(1).collapse()
V_uz, _ = V.sub(2).collapse()

left_dofs = fem.locate_dofs_geometrical(
    (V.sub(0), V_ux), lambda x: np.isclose(x[0], 0.0)
)
back_dofs = fem.locate_dofs_geometrical(
    (V.sub(1), V_uy), lambda x: np.isclose(x[1], 0.0)
)
bot_dofs = fem.locate_dofs_geometrical(
    (V.sub(2), V_uz), lambda x: np.isclose(x[2], 0.0)
)
top_dofs = fem.locate_dofs_geometrical(
    (V.sub(2), V_uz), lambda x: np.isclose(x[2], height)
)

uD_x0 = fem.Function(V_ux)
uD_y0 = fem.Function(V_uy)
uD_z0 = fem.Function(V_uz)
uD_z1 = fem.Function(V_uz)


uD_z1.vector.set(0.0)

bcs = [
    fem.dirichletbc(uD_x0, left_dofs, V.sub(0)),
    fem.dirichletbc(uD_y0, back_dofs, V.sub(1)),
    fem.dirichletbc(uD_z0, bot_dofs,  V.sub(2)),
    fem.dirichletbc(uD_z1, top_dofs,  V.sub(2))
]


# -----------------------------------------------------------------------------------
# DEFINE QUADRATURE ELEMENTS
# -----------------------------------------------------------------------------------
deg_quad = 2  # quadrature degree for internal state variable representation
vdim = 6  # dimension of the vectorial representation of tensors

resistance_len = 12
basic_tensor_len = 3

W0e = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad
)
We = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(vdim,), scheme="default", degree=deg_quad
)
We_2 = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(resistance_len,), scheme="default", degree=deg_quad
)
WTe = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(vdim,vdim), scheme="default", degree=deg_quad
)
WTe_2 = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(basic_tensor_len,basic_tensor_len), scheme="default", degree=deg_quad
)

W = fem.functionspace(domain, We)        #For stress and strain in vector format 6x1
W_2 = fem.functionspace(domain, We_2)    #For resistance 12x1
WT = fem.functionspace(domain, WTe)      #For tangent stiffness 6x6
WT_2 = fem.functionspace(domain, WTe_2)  #For Fp and Lp tensors 3x3
W0 = fem.functionspace(domain, W0e)      #For constants


# -----------------------------------------------------------------------------------
# DEFINE VARIATIONAL FORMULATION
# -----------------------------------------------------------------------------------

#############################################################################
# This small part is for assigning the crystal orientations
orien_function = fem.Function(DG0, name="Orientations")
orien_function.x.array[:] = create_element_to_tag_map(domain, cell_tags)
# Create a function on the quadrature space
orien_tags = fem.Function(W0)
# Get the dofmap for the quadrature space
dofmap = W0.dofmap
# Iterate over cells, assign cell numbers to dofs, and assign tags to quadrature points
for cell in range(domain.topology.index_map(domain.topology.dim).size_local):
    dofs = dofmap.cell_dofs(cell)
    cell_tag = orien_function.x.array[cell]
    for dof in dofs:
        orien_tags.x.array[dof] = cell_tag
# Write xdmf file for visualization on paraview
out_file = "crystal_plasticity.xdmf"
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(orien_function)
#############################################################################


sig = fem.Function(W, name="Stress")
Ct = fem.Function(WT, name="Tangent_operator")

Fp_old = fem.Function(WT_2)
Fp_old_temp = fem.Function(WT_2)
# Assign an initial value to the plastic deformation gradient
# Get the number of DOFs
num_dofs = Fp_old.vector.getSize()
# Create an array of identity matrices
identity_values = np.tile(np.eye(3), (num_dofs // 9, 1, 1)).flatten()
Fp_old.x.array[:] = identity_values
Fp_old_temp.x.array[:] = Fp_old.x.array

Lp_old = fem.Function(WT_2)
Lp_old_temp = fem.Function(WT_2)

s_0 = mat_prop['yield_resistance']
resist = fem.Function(W_2)
resist_temp = fem.Function(W_2)
# Assign an initial value to the resistance
resist.interpolate(lambda x: s_0 * np.ones((12, x.shape[1])))
resist_temp.interpolate(lambda x: s_0 * np.ones((12, x.shape[1])))


u = fem.Function(V, name="Total_displacement")
Ddu = fem.Function(V, name="Acumulated_increments")
Du = fem.Function(V, name="Current_increment")

u_ = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

Id = ufl.Identity(3)


F = ufl.variable(Id + ufl.grad(u))

def strain_increment(del_u):
    grad_del_u = ufl.grad(del_u)
    D_E = ufl.sym(ufl.dot(ufl.transpose(grad_del_u),F))
    return ufl.as_tensor([D_E[0, 0], 
                          D_E[1, 1], 
                          D_E[2, 2],
                          np.sqrt(2) * D_E[1, 2],
                          np.sqrt(2) * D_E[0, 2],
                          np.sqrt(2) * D_E[0, 1]])

def strain_variation(v):
    grad_v = ufl.grad(v)
    d_E = ufl.sym(ufl.dot(ufl.transpose(grad_v),F))
    return ufl.as_tensor([d_E[0, 0], 
                          d_E[1, 1], 
                          d_E[2, 2],
                          np.sqrt(2) * d_E[1, 2],
                          np.sqrt(2) * d_E[0, 2],
                          np.sqrt(2) * d_E[0, 1]])

def incr_strain_variation(del_u, v):
    grad_v = ufl.grad(v)
    grad_del_u = ufl.grad(del_u)
    D_d_E = ufl.sym(ufl.dot(ufl.transpose(grad_v),grad_del_u))
    return ufl.as_tensor([D_d_E[0, 0], 
                          D_d_E[1, 1], 
                          D_d_E[2, 2],
                          np.sqrt(2) * D_d_E[1, 2],
                          np.sqrt(2) * D_d_E[0, 2],
                          np.sqrt(2) * D_d_E[0, 1]])


dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"},
)



Residual = ufl.dot(sig, strain_variation(v)) * dx
tangent_form = (ufl.dot(ufl.dot(strain_variation(v), Ct), strain_increment(u_)) + ufl.dot(sig, incr_strain_variation(u_, v)))* dx



# -----------------------------------------------------------------------------------
# FUNCTION FOR EVALUATE AT QUADRATURE POINTS
# -----------------------------------------------------------------------------------
basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)

map_c = domain.topology.index_map(domain.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
cells = np.arange(0, num_cells, dtype=np.int32)
ngauss = num_cells * len(weights)

eps_expr = fem.Expression(F, quadrature_points)


def eval_at_quadrature_points(expression):
    return expression.eval(domain, cells)#.reshape(ngauss, -1)


# -----------------------------------------------------------------------------------
# DEFINE SOLVER
# -----------------------------------------------------------------------------------
class CustomLinearProblem(fem.petsc.LinearProblem):
    def assemble_rhs(self, u=None):
        """Assemble right-hand side and lift Dirichlet bcs.

        Parameters
        ----------
        u : dolfinx.fem.Function, optional
            For non-zero Dirichlet bcs u_D, use this function to assemble rhs with the value u_D - u_{bc}
            where u_{bc} is the value of the given u at the corresponding. Typically used for custom Newton methods
            with non-zero Dirichlet bcs.
        """

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        x0 = [] if u is None else [u.vector]
        fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=x0, scale=1.0)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        x0 = None if u is None else u.vector
        fem.petsc.set_bc(self._b, self.bcs, x0, scale=1.0)

        

    def assemble_lhs(self):
        self._A.zeroEntries()
        fem.petsc.assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

    def solve_system(self):
        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()


# -----------------------------------------------------------------------------------
# MATERIAL CONSTITUTIVE BEHAVIOR (JAX-SIDE)
# -----------------------------------------------------------------------------------
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


def plastic_deformation_gradient(Lp,Fp_old,delta_t):
    inverse_part = (I2 - delta_t * Lp)
    Fp_new = jnp.linalg.solve(inverse_part, Fp_old)
    return Fp_new


def elastic_deformation_gradient(F,Fp):
    Fe = jnp.matmul(F , jnp.linalg.inv(Fp))
    return Fe


def resistence_integration(s_alpha, s_alpha_dot,delta_t):
    s_alpha_new = s_alpha + s_alpha_dot*delta_t
    return s_alpha_new


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

    product_tensor = jnp.matmul(second_piola , right_cauchy)

    # Perform the double contraction with the product tensor for each slip system
    rShears = double_contraction_3o_2o(P0_sn, product_tensor)

    return rShears


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
    Fe_T = Fe.mT

    # Calculando dCe_dFe sumando los productos tensoriales adecuados
    dCe_dFe = double_contraction_4o_4o(tensor_product_2o_2o(I2,Fe), I4_t) + tensor_product_2o_2o(Fe_T, I2)

    # Calculando dS_dFe utilizando la contracción doble
    dSe_dFe = double_contraction_4o_4o(D4, 0.5 * dCe_dFe)

    # Calculando dA_dFe usando las multiplicaciones definidas
    dA_dFe = double_contraction_4o_4o(tensor_product_2o_2o(I2,Ce),dSe_dFe) + double_contraction_4o_4o(tensor_product_2o_2o(S,I2),dCe_dFe)

    # Calculando dFe_dLp_trial
    dFe_dLp_trial = -delta_t * double_contraction_4o_4o(tensor_product_2o_2o(F,I2), tensor_product_2o_2o(jnp.linalg.inv(Fp),I2))

    # Perform double inner product
    dA_dLp_trial = double_contraction_4o_4o(dA_dFe, dFe_dLp_trial)

    return dA_dLp_trial


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
    - dLp_dA:  Derivative of the velocity gradient with respect to stress, this is re-used on the system tangent stiffness
    """
    dLp_dA = derivative_Lp_wrt_A(tau, P0_sn, gamma_dot_0, resistance, m)

    dA_dLp_trial = derivative_A_wrt_Lp_trial(Fe, S, Ce, delta_t, Fp, F, D4)

    dLp_dLp_trial = double_contraction_4o_4o(dLp_dA, dA_dLp_trial)

    dR_dLp_trial = fourth_order_identity() - dLp_dLp_trial

    return dR_dLp_trial, dLp_dA


def compute_U(Cc):
    # Compute eigenvalues and eigenvectors of C
    eigvals, eigvecs = eigh(Cc)

    # Compute principal strains
    principal_strains = jnp.sqrt(eigvals)

    # Compute U (right stretch tensor)
    U = jnp.einsum('i,ji,ki->jk', principal_strains, eigvecs, eigvecs)
    return U


def compute_component(eigvecs, i, j, weight, n):
    # Compute n_i ⊗ n_j ⊗ n_j ⊗ n_i
    outer_product = jnp.outer(eigvecs[:, i], eigvecs[:, j])
    return weight * jnp.outer(outer_product, outer_product).reshape((n, n, n, n))

compute_component_vmap = jax.vmap(jax.vmap(compute_component, in_axes=(None, None, 0, 0, None)), in_axes=(None, 0, None, None, None))

def compute_dU_dC_manual(Cc):
    # Compute eigenvalues and eigenvectors of C
    eigvals, eigvecs = eigh(Cc)
    
    # Compute principal strains
    principal_strains = jnp.sqrt(eigvals)
    
    n = len(eigvals)
    
    # Pre-compute weights matrix
    # Compute 1 / (λ_i^(1/2) + λ_j^(1/2))
    weights = 1.0 / (principal_strains[:, None] + principal_strains[None, :])
    
    # Use vmap to compute all components at once
    # Compute n_i ⊗ n_j ⊗ n_j ⊗ n_i
    tensor = compute_component_vmap(eigvecs, jnp.arange(n), jnp.arange(n), weights, n)
    
    return jnp.sum(tensor, axis=(0, 1))


# Compute the Jacobian (derivative) of U with respect to C
compute_dU_dC = jax.jacfwd(compute_U)


# Function to check if a matrix is NaN and replace if necessary
def check_and_replace_with_elastic(K_mat,D4):
    # Compute the norm and check if it's NaN
    norm_is_nan = jnp.isnan(jnp.linalg.norm(K_mat))
    
    # Use jax.lax.cond to select the appropriate tensor
    K_mat_valid = jax.lax.cond(norm_is_nan,
                               lambda: D4,  # If norm is NaN, return elastic tensor
                               lambda: K_mat)  # Else, return the original K_mat
    
    return K_mat_valid


def material_tang(F, Fp, Se, del_t, Fp0, D4, dLp_dA):
 
    #SImple tensor calculations
    Fp_inv = jnp.linalg.inv(Fp)
    Fe = elastic_deformation_gradient(F, Fp)
    Fe_t = Fe.mT
    Ce = jnp.matmul(Fe_t , Fe)
    
    C = jnp.matmul(F.mT , F)

    Fp0_inv = jnp.linalg.inv(Fp0)

    U = compute_U(C)
    R = jnp.matmul(F , jnp.linalg.inv(U))

    dF_dU = simple_contraction_2o_4o(R,I4)#tensor_product_2o_2o(R,I2)

    dU_dC = compute_dU_dC_manual(C)

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
    dS_dF_part2 = tensor_product_2o_2o(I2, simple_contraction_2o_2o(Se,Fp_inv.mT))
    dS_dF_part3 = tensor_product_2o_2o(simple_contraction_2o_2o(Fp_inv,Se),I2)
    dS_dF = double_contraction_4o_4o(dS_dF_part1,dSe_dF) + double_contraction_4o_4o(dS_dF_part2,dFp_inv_dF) + double_contraction_4o_4o(double_contraction_4o_4o(dS_dF_part3,I4_t),dFp_inv_dF)

    dS_dE = double_contraction_4o_4o(dS_dF, dF_dE)

    return dS_dE


def material_model_jit(F, Fp_prev, Lp_prev, gp_orient, resistance, del_time):
    '''
    Function to compute all the quantities related to the material model
    
    In:
    - F: Deformation gradient.
    - Fp_prev: Previous plastic deformation gradient
    - Lp_prev: Plastic velocity gradient (vectorial format)
    - gp_orient: Orientation tag for the corresponding schmidt tensor
    - Resistance: Resistance of all the slip systems
    - del_time: time step
    
    Out:
    - S: Second Piola Kirchhoff
    - Fp: Plastic deformation gradient of the current step
    - Lp_trial: Velocity gradient of the current step
    - new_resistance: Updated slip resistances
    '''
    # Transform Fp, F and Lp to 3x3 format
    F = as_3x3_tensor(F)
    Fp_prev = as_3x3_tensor(Fp_prev)
    Lp_prev = as_3x3_tensor(Lp_prev)

    gamma_dot_0 = mat_prop['slip_rate']
    s_inf = mat_prop['saturation_strenght']
    h_0 = mat_prop['hardening_slope']
    m = mat_prop['exponent_m']
    a = mat_prop['exponent_a']
    q = mat_prop['q']
    
    D4 = mat_prop["Elastic_tangent"]#fourth_order_elasticity(E, nu)
    
    P0_sn = P0_sn_list[gp_orient.astype(int)-1]

    def body_fn(state):
        iteration, converged, _, _, Lp_trial, _, _ = state

        # Plastic deformation gradient
        Fp = plastic_deformation_gradient(Lp_trial, Fp_prev, del_time)
        
        # Elastic deformation gradient
        Fe = elastic_deformation_gradient(F, Fp)
        
        # Right Cauchy-Green strain
        r_cauchy = jnp.matmul(Fe.mT , Fe)

        
        # Green lagrange strain
        g_lagrange = 0.5 * (r_cauchy - I2)
        
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
        jacobian_R_L, dLp_dA = derivative_R_wrt_Lp_trial(tau, P0_sn, gamma_dot_0, resistance, m, Fe, Se, r_cauchy, del_time, Fp_prev, F, D4)
        # correction = double_contraction_4o_2o(invert_4o_tensor(jacobian_R_L), Residual)
        correction = jax.scipy.linalg.solve(jacobian_R_L.reshape(9, 9) , Residual.reshape(9))
        Lp_trial = Lp_trial - correction.reshape(3,3)
        
        iteration = iteration + 1

        return iteration, converged, Se, Fp, Lp_trial, gamma_dot, dLp_dA
 
    def cond_fn(state):
        iteration, converged, _, _, _, _, _ = state
        return jnp.logical_and(iteration < 1000, jnp.logical_not(converged))
    
    # Initialize the state with dummy values
    initial_state = (0, False, jnp.zeros((3, 3)), Fp_prev, Lp_prev, jnp.zeros(len(resistance)), jnp.zeros((3,3,3,3)))
    
    # Run the while loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    iteration, _, Se, Fp, Lp_trial, gamma_dot, tang = final_state

    # Calculate the tangent stiffness
    tang = material_tang(F, Fp, Se, del_time, Fp_prev,D4, tang)

    # Calculate the non-elastic 2-PK
    S2pk = jnp.matmul(jnp.matmul(jnp.linalg.inv(Fp) , Se) , jnp.linalg.inv(Fp.mT))

    # Actualize for the new resistance after convergence
    s_dot = slip_resistance_rate(gamma_dot, h_0, resistance, s_inf, a, q)
    new_resistance = resistence_integration(resistance, s_dot, del_time)
    
    # Return stress and tangent stiffness in Mandel format
    stress = mandel_2o_to_vector(S2pk)
    tangent = mandel_4o_to_matrix(tang)

    return stress, Fp, Lp_trial, new_resistance, tangent


# -----------------------------------------------------------------------------------
# JIT AND VECTORIZATION
# -----------------------------------------------------------------------------------
def ravel_results(S2pk, Fp, Lp_trial, new_resistance, K_mat):
    return S2pk.ravel(), Fp.ravel(), Lp_trial.ravel(), new_resistance.ravel(), K_mat.ravel()

@jax.jit
def batched_constitutive_update(F_val, Fp_prev_val, Lp_prev_val, gp_orient, resist_val, del_time):
    # get the number of Gauss points
    n_gp = len(Fp_prev_val) // 9

    # Reshape some inputs Gauss-Point-wise before vectorize
    Fp_prev_val = Fp_prev_val.reshape(n_gp,-1)
    F_val = F_val.reshape(n_gp,-1)
    Lp_prev_val = Lp_prev_val.reshape(n_gp,-1)
    resist_val = resist_val.reshape(n_gp,-1)

    # Apply vmap over Gauss-points to batch-process material_model_jit
    results = jax.vmap(material_model_jit, in_axes=(0, 0, 0, 0, 0, None))(
        F_val, Fp_prev_val, Lp_prev_val, gp_orient, resist_val, del_time
    )
    
    # Unpack the results
    S2pk, Fp, Lp_trial, new_resistance, K_mat = results
    
    # Ravel the results
    return ravel_results(S2pk, Fp, Lp_trial, new_resistance, K_mat)


# -----------------------------------------------------------------------------------
# CONSTITUTIVE UPDATE FUNCTION(JAX-DOLFINX COMUNICATION)
# -----------------------------------------------------------------------------------
def constitutive_update(u, sig, Fp_prev, Lp_prev, resist, del_time):
    with Timer("Constitutive update"):
        # Evaluate displacement
        F_val = eval_at_quadrature_points(eps_expr)

        # Gauss Point corresponding orientations
        gp_orient = orien_tags.x.array

        # Reshape state parameters
        Fp_prev_val = Fp_prev.x.array#.reshape(ngauss, -1)
        Lp_prev_val = Lp_prev.x.array#.reshape(ngauss, -1)
        resist_val = resist.x.array#.reshape(ngauss, -1)

        # Call the constitutive law
        sig.x.array[:], Fp_old_temp.x.array[:], Lp_old_temp.x.array[:], resist_temp.x.array[:], Ct.x.array[:] = batched_constitutive_update(
            F_val, Fp_prev_val, Lp_prev_val, gp_orient, resist_val, del_time
        )
    return True


# -----------------------------------------------------------------------------------
# DEFINE LINEAR PROBLEM AND SOLVER
# -----------------------------------------------------------------------------------
tangent_problem = CustomLinearProblem(
    tangent_form,
    -Residual,
    u=Du,
    bcs=bcs,
    petsc_options={
        "ksp_type": "gmres", #gmres and cg take the same time
        "pc_type": "ilu",
        "ksp_rtol": 1e-6,
        "ksp_max_it": 1000
    },  
)


# -----------------------------------------------------------------------------------
# TIME-STEPPING LOOP
# 850 iterations
# -----------------------------------------------------------------------------------
Nitermax, tol = 200, 1e-6  # parameters of the Newton-Raphson procedure
Nincr = 200
ttime = 200
results = np.zeros((Nincr, 3))
del_time = ttime/Nincr

s_0 = mat_prop['yield_resistance']

# # This is temporal, just for storing F values to compare with standalone material model
# deformation_gradients = np.zeros((Nincr, 9))

# we set all functions to zero before entering the loop in case we would like to reexecute this code cell
sig.vector.set(0.0)
Lp_old_temp.vector.set(0.0)

u.vector.set(0.0)
Du.vector.set(0.0)
Ddu.vector.set(0.0)

new_stretch = 0.0
stretch_max = height/1000 # 0.001

# Run the first time to update sigma Ct and state parameters so the rhs and lhs could be assembled
check = constitutive_update(u, sig, Fp_old, Lp_old, resist, del_time)
# deformation_gradients[0,:] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

total_NR_counter = 0
# for i in range(1,Nincr):
for i in range(1,40):
    # Apply the boundary condition for this load step
    new_stretch = stretch_max/Nincr   # 5e-6 steps

    # Apply boundary conditions
    uD_z1.vector.set(new_stretch)

    # Reset the acumulated increment for this step
    Ddu.x.array[:] = 0.0

    # compute the residual norm at the beginning of the load step
    tangent_problem.assemble_rhs()
    nRes0 = tangent_problem._b.norm()
    nRes = nRes0

    niter = 0
    while nRes / nRes0 > tol and niter < Nitermax:
        print("antes del sistema")
        # solve for the displacement correction
        tangent_problem.assemble_lhs()
        tangent_problem.solve_system() # After this Du has a value for diaplacement

        # update Ddu used to store all the time-step displacmeent increments
        Ddu.vector.axpy(1, Du.vector)  # Ddu = Ddu + 1*Du
        Ddu.x.scatter_forward()

        # update the displacement with the current correction
        u.vector.axpy(1, Du.vector)  # u = u + 1*Du
        u.x.scatter_forward()
        print("despues del sistema")

        # Recalculate sigma Ct and state parameters (THIS IS CONSIDERING Du)
        check = constitutive_update(u, sig, Fp_old, Lp_old, resist, del_time)
        # deformation_gradients[i,:] = F_mean

        print("despues del constitutive_update")


        # Lift RHS considering all the increments of this load-step in the dirichlet bc
        tangent_problem.assemble_rhs(Ddu)

        nRes = tangent_problem._b.norm()

        sig_values = sig.x.array.reshape(ngauss, -1)
        print("--error: ",nRes / nRes0)

        print("new_stretch is: ", new_stretch)
        max_u = np.max(np.abs(u.x.array))
        print(f"For iteration {niter}, max |u|: {max_u}")
        niter += 1

    # Update the state parameters
    resist.x.array[:] = resist_temp.x.array[:]
    Fp_old.x.array[:] = Fp_old_temp.x.array[:]
    Lp_old.x.array[:] = Lp_old_temp.x.array[:]

    # Post-processing
    sig_values = sig.x.array.reshape(ngauss, -1)

    with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
        xdmf.write_function(u, i+1)

    results[i, 0] = max_u
    results[i, 1] = np.mean(sig_values[:, 2])
    results[i, 2] = niter

    print(f"-----------Iteration {niter}----------Step {i} done!-----------")
    total_NR_counter = total_NR_counter + niter

# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time
print(f"Time spent: {elapsed_time} seconds")
print(f"Total Newton-Raphson iterations: {total_NR_counter}")

# -----------------------------------------------------------------------------------
# PLOT RESULTS
# -----------------------------------------------------------------------------------

plt.figure()
plt.plot(results[:, 0],results[:, 1], "-oC3", label="FEM")
plt.xlabel(r"Strain $\varepsilon_{zz}$")
plt.ylabel(r"Stress $\sigma_{zz}$ [MPa]")
plt.legend()
plt.grid()
plt.show()
plt.savefig('plot1_rve.png')

np.savetxt('results.txt', results, delimiter=',')
# np.savetxt('deformation_gradients.txt', deformation_gradients, delimiter=',')


# plt.figure()
# plt.bar(np.array(range(0,201)), results[:, 2])
# plt.xlabel("Time step")
# plt.ylabel("Number of iterations")
# plt.show()