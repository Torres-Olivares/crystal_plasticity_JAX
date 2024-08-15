from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
import basix
import numpy as np

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
tag_function = fem.Function(DG0)
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






# def calculate_schmidt_tensor(orientation):
#     # Implement your Schmidt tensor calculation here
#     # This is just a placeholder function
#     return np.random.rand(12, 3)  # Replace with actual calculation

# orientations = [
#     [0, 0, 0],
#     [45, 0, 0]
#     # Add more orientations as needed
# ]

# schmidt_tensors = np.array([calculate_schmidt_tensor(ori) for ori in orientations])


# print(schmidt_tensors)