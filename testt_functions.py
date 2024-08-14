# def calculate_schmidt_tensor(orientation):
#     # Implement your Schmidt tensor calculation here
#     # This is just a placeholder function
#     return np.random.rand(12, 3)  # Replace with actual calculation

# orientations = [
#     [0, 0, 0],
#     [45, 0, 0],
#     [0, 45, 0],
#     # Add more orientations as needed
# ]

# schmidt_tensors = np.array([calculate_schmidt_tensor(ori) for ori in orientations])


# print(schmidt_tensors)

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
import ufl
import basix
import dolfinx.fem.petsc
from petsc4py import PETSc


from dolfinx import mesh, fem, io

import gmsh

hsize = 0.2 # Mesh size

# Square dimensions
side_length = 1.0

# Initialize Gmsh
gmsh.initialize()
gdim = 2
model_rank = 0

if MPI.COMM_WORLD.rank == 0:
    gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
    gmsh.model.add("Model")

    geom = gmsh.model.geo

    # Define the corner points of the square
    p1 = geom.add_point(0, 0, 0)
    p2 = geom.add_point(side_length, 0, 0)
    p3 = geom.add_point(side_length, side_length, 0)
    p4 = geom.add_point(0, side_length, 0)

    # Define the lines between points to form the square
    l1 = geom.add_line(p1, p2)
    l2 = geom.add_line(p2, p3)
    l3 = geom.add_line(p3, p4)
    l4 = geom.add_line(p4, p1)

    # Create a curve loop and surface
    square_boundary = geom.add_curve_loop([l1, l2, l3, l4])
    square_surface = geom.add_plane_surface([square_boundary])

    # Synchronize the geometry
    geom.synchronize()

    # Define mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)

    # Set meshing options for structured quadrilateral grid
    gmsh.option.setNumber("Mesh.Algorithm", 11)  # Use Delaunay triangulation (1 for triangulation, 2 for quads)


    # Add physical groups (optional, useful for boundary conditions)
    # gmsh.model.addPhysicalGroup(gdim, [square_surface], 1)
    gmsh.model.addPhysicalGroup(gdim - 1, [l1], 1, name="bottom")
    gmsh.model.addPhysicalGroup(gdim - 1, [l2], 2, name="right")
    gmsh.model.addPhysicalGroup(gdim - 1, [l3], 3, name="top")
    gmsh.model.addPhysicalGroup(gdim - 1, [l4], 4, name="left")

    # Generate the mesh
    gmsh.model.mesh.generate(gdim)

    # Save the mesh to a .msh file
    msh_file = "CP_orientations.msh"
    gmsh.write(msh_file)


# Convert to DOLFINx mesh
domain, _, facets = io.gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim
)
gmsh.finalize()



# Write mesh to an XDMF file
out_file = "CP_orientations.xdmf"
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)






# import meshio

# # Read the .msh file
# mesh = meshio.read("n10-id1.msh")



# # Write the mesh to a VTK file
# meshio.write("your_file.vtk", mesh)