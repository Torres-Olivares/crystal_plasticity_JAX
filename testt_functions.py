from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, io
import gmsh
import numpy as np


# Assuming your file is named "your_mesh_file.msh"
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("neper_files/n2-id1.msh", MPI.COMM_WORLD)
poly1_entities = cell_tags.find(1)
poly2_entities = cell_tags.find(2)


# Create a function space for the cell tags
DG0 = fem.functionspace(domain, ("DG", 0))
tag_function = fem.Function(DG0, name="Grain")

# Initialize the tag_function with zeros
tag_function.x.array[:] = 0

# Set the values for the tagged cells
tag_function.x.array[poly1_entities] = 1
tag_function.x.array[poly2_entities] = 2


# Write mesh to an XDMF file
out_file = "CP_orientations.xdmf"
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(tag_function)




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