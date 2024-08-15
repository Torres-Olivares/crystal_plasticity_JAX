from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
import basix
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R
from jax.scipy.linalg import eigh

# Read the mesh
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("neper_files/n2-id1.msh", MPI.COMM_WORLD)

dim = domain.topology.dim
print(f"Mesh topology dimension d={dim}.")

# Create anumpy array containing the orientations per element
def create_element_to_tag_map(domain, cell_tags):
    element_to_tag = np.zeros(domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32)
    for tag in np.unique(cell_tags.values):
        cells = cell_tags.find(tag)
        element_to_tag[cells] = tag
    return element_to_tag


# Create tag function on DG0 space
DG0 = fem.functionspace(domain, ("DG", 0))
tag_function = fem.Function(DG0, name="Orientations")
tag_function.x.array[:] = create_element_to_tag_map(domain, cell_tags)

# Define quadrature function space
degree = 2  # or whatever degree you're using
Qe = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(), scheme="default", degree=degree
)
Q = fem.functionspace(domain, Qe)

# Create a function on the quadrature space
quad_tags = fem.Function(Q)

# Get the dofmap for the quadrature space
dofmap = Q.dofmap

# Iterate over cells, assign cell numbers to dofs, and assign tags to quadrature points
for cell in range(domain.topology.index_map(domain.topology.dim).size_local):
    dofs = dofmap.cell_dofs(cell)
    cell_tag = tag_function.x.array[cell]
    for dof in dofs:
        quad_tags.x.array[dof] = cell_tag


print(quad_tags.x.array)


# Write results to XDMF
out_file = "CP_orientations.xdmf"
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(tag_function)




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

# Initial s and n vectors
sl_0, nl_0 = slip_systems_fcc()


def calculate_schmidt_tensor(initial_angles):
    # Implement your Schmidt tensor calculation here
    rot_euler = R.from_euler('ZXZ', initial_angles, degrees=True)
    matrix = rot_euler.as_matrix()
    new_sl0 = sl_0 @ matrix.T
    new_nl0 = nl_0 @ matrix.T
    return jnp.einsum('bi,bj->bij', new_sl0, new_nl0)

orientations = [
    [0, 0, 0],
    [45, 0, 0]
]


schmidt_tensors = jnp.array([calculate_schmidt_tensor(ori) for ori in orientations])


print(schmidt_tensors[0])