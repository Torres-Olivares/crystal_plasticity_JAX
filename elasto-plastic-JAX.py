# -----------------------------------------------------------------------------------
# IMPORT THE REQUIERED LIBRARIES
# -----------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

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

import time

jax.config.update("jax_enable_x64", True)  # use double-precision

# Record the start time
start_time = time.time()
# -----------------------------------------------------------------------------------
# DEFINE MESH AND FUNCTIONSPACE
# -----------------------------------------------------------------------------------
length, height = 1, 1
N = 20
domain = create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [length, length, height]],
    [N, N, N],
    CellType.hexahedron,
)

dim = domain.topology.dim
print(f"Mesh topology dimension d={dim}.")

degree = 1
shape = (dim,)
V = fem.functionspace(domain, ("P", degree, shape))



out_file = "elasto-plastic2.xdmf"
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)



##############just for plotting#####################
# Because mesh is linear and V is quadratric
# V_epsv = fem.functionspace(domain, ("P", degree, (3,)))
# epsv_proj = fem.Function(V_epsv, name="Viscous_strain_projected")
# ####################################################

# -----------------------------------------------------------------------------------
# DEFINE MATERIAL PROPERTIES
# -----------------------------------------------------------------------------------
mat_prop = {
    "YoungModulus": 70e3,
    "PoissonRatio": 0.3, 
    "Yield_strength": 300.0,
    "Tangent": 70e3 / 100.0
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

# uD_y = fem.Function(V_uy)
epsr = 1e-3
# uD_z1.vector.set(epsr * height)
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

W0e = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad
)
We = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(vdim,), scheme="default", degree=deg_quad
)
WTe = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(vdim,vdim), scheme="default", degree=deg_quad
)
W = fem.functionspace(domain, We)
WT = fem.functionspace(domain, WTe)
W0 = fem.functionspace(domain, W0e)


# -----------------------------------------------------------------------------------
# DEFINE VARIATIONAL FORMULATION
# -----------------------------------------------------------------------------------
sig = fem.Function(W, name="Stress")
Ct = fem.Function(WT, name="Tangent_operator")
# eps_old = fem.Function(W, name="Previous_total_strain")

sig_old = fem.Function(W)
p = fem.Function(W0, name="Cumulative_plastic_strain")
dp = fem.Function(W0)

u = fem.Function(V, name="Total_displacement")
du = fem.Function(V, name="Iteration_correction")
Du = fem.Function(V, name="Current_increment")

v = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

def eps_Mandel(v):
    e = ufl.sym(ufl.grad(v))
    return ufl.as_tensor([e[0, 0], 
                          e[1, 1], 
                          e[2, 2],
                          np.sqrt(2) * e[1, 2],
                          np.sqrt(2) * e[0, 2],
                          np.sqrt(2) * e[0, 1]])

dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"},
)

Residual = ufl.dot(eps_Mandel(u_), sig) * dx
tangent_form = ufl.dot(eps_Mandel(v), ufl.dot(Ct, eps_Mandel(u_))) * dx


# -----------------------------------------------------------------------------------
# FUNCTION FOR EVALUATE AT QUADRATURE POINTS
# -----------------------------------------------------------------------------------
basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)

map_c = domain.topology.index_map(domain.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
cells = np.arange(0, num_cells, dtype=np.int32)
ngauss = num_cells * len(weights)

eps_expr = fem.Expression(eps_Mandel(Du), quadrature_points)

def eval_at_quadrature_points(expression):
    return expression.eval(domain, cells).reshape(ngauss, -1)


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

def as_3D_tensor(X):
    return jnp.array([[               X[0], np.sqrt(1/2) * X[5], np.sqrt(1/2) * X[4]],
                      [np.sqrt(1/2) * X[5],                X[1], np.sqrt(1/2) * X[3]],
                      [np.sqrt(1/2) * X[4], np.sqrt(1/2) * X[3],                X[2]]])

def to_vect(X):
    return jnp.array([X[0, 0], 
                      X[1, 1], 
                      X[2, 2], 
                      np.sqrt(2) * X[1, 2], 
                      np.sqrt(2) * X[0, 2], 
                      np.sqrt(2) * X[0, 1]])

def ppos(x):
    return jnp.maximum(x, 0)

def dev(X):
    return X - jnp.trace(X) / 3 * jnp.eye(3)

def local_constitutive_update(delta_eps, old_sig, old_p):

    E0 = mat_prop["YoungModulus"]
    nu = mat_prop["PoissonRatio"]
    sig0 = mat_prop["Yield_strength"]
    Et = mat_prop["Tangent"]
    H = E0 * Et / (E0 - Et)

    lmbda = E0 * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E0 / (2 * (1 + nu))

    sig_n = as_3D_tensor(old_sig)
    del_eps = as_3D_tensor(delta_eps)
    sig_elastic = lmbda * jnp.trace(del_eps) * jnp.eye(3) + 2 * mu * del_eps
    sig_elas = sig_n + sig_elastic
    s = dev(sig_elas)
    sig_eq = jnp.sqrt(3 / 2.0 * jnp.sum(s * s))
    f_elas = sig_eq - sig0 - H * old_p
    dp = ppos(f_elas) / (3 * mu + H)

    factor = 1e-10  # Small number to avoid division by zero
    n_elas = jnp.where(sig_eq > factor, 
                       jnp.where(jnp.abs(f_elas) > factor, 
                                 s / sig_eq * ppos(f_elas) / f_elas, 
                                 jnp.zeros_like(s)),
                       jnp.zeros_like(s))

    # Prevent division by zero for beta calculation
    beta = jnp.where(sig_eq > factor, 3 * mu * dp / sig_eq, 0.0)

    new_sig = sig_elas - beta * s
    
    sig_vec = to_vect(new_sig)

    state = (sig_vec, dp)

    return sig_vec, state


# This derivative returns (tangent, (stress, viscous-strain))
tangent_operator_and_state = jax.jacfwd(
    local_constitutive_update, argnums=0, has_aux=True
)


# -----------------------------------------------------------------------------------
# JIT AND VECTORIZATION
# -----------------------------------------------------------------------------------
batched_constitutive_update = jax.jit(
    jax.vmap(tangent_operator_and_state, in_axes=(0, 0, 0))
)


# -----------------------------------------------------------------------------------
# CONSTITUTIVE UPDATE FUNCTION(JAX-DOLFINX COMUNICATION)
# -----------------------------------------------------------------------------------
def constitutive_update(Du, sig, sigm_old, p_old):
    with Timer("Constitutive update"):
        deps_values = eval_at_quadrature_points(eps_expr)

        sig_old_values = sigm_old.x.array.reshape(ngauss, -1)
        p_old_values = p_old.x.array.reshape(ngauss, -1)

        Ct_values, state = batched_constitutive_update(
            deps_values, sig_old_values, p_old_values
        )
        sig_values, dp_values = state

        sig.x.array[:] = sig_values.ravel()
        dp.x.array[:] = dp_values.ravel()
        Ct.x.array[:] = Ct_values.ravel()


# -----------------------------------------------------------------------------------
# DEFINE LINEAR PROBLEM AND SOLVER
# -----------------------------------------------------------------------------------
tangent_problem = CustomLinearProblem(
    tangent_form,
    -Residual,
    u=du,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)


# -----------------------------------------------------------------------------------
# TIME-STEPPING LOOP
# -----------------------------------------------------------------------------------
Nitermax, tol = 200, 1e-6  # parameters of the Newton-Raphson procedure
Nincr = 40
results = np.zeros((Nincr + 1, 3))

# we set all functions to zero before entering the loop in case we would like to reexecute this code cell
sig.vector.set(0.0)
sig_old.vector.set(0.0)
p.vector.set(0.0)
u.vector.set(0.0)
Du.vector.set(0.0)

new_stretch = 0.0
stretch_max = height/40

# Run the first time to update sig and Ct so the rhs and lhs could be assembled
constitutive_update(Du, sig, sig_old, p)

for i in range(1,41):
    # Apply the boundary condition for this load step
    new_stretch = new_stretch + stretch_max/Nincr

    # Apply boundary conditions
    uD_z1.vector.set(stretch_max/Nincr)

    # Reset the increment for this load step
    Du.x.array[:] = 0.0

    # compute the residual norm at the beginning of the load step
    tangent_problem.assemble_rhs()
    nRes0 = tangent_problem._b.norm()
    nRes = nRes0

    niter = 0
    while nRes / nRes0 > tol and niter < Nitermax:
        # solve for the displacement correction
        tangent_problem.assemble_lhs()
        tangent_problem.solve_system() # After this du has a value for diaplacement

        # update the displacement increment with the current correction
        Du.vector.axpy(1, du.vector)  # Du = Du + 1*du
        Du.x.scatter_forward()

        # Recalculate sigma Ct and p (THIS IS CONSIDERING Du)
        constitutive_update(Du, sig, sig_old, p)

        # compute the new residual
        tangent_problem.assemble_rhs(Du)
        nRes = tangent_problem._b.norm()
        print(f"For iteration {niter}, max |Du|: {np.max(np.abs(Du.x.array))}")
        niter += 1

    # Update the displacement with the converged increment
    u.vector.axpy(1, Du.vector)  # u = u + 1*Du
    u.x.scatter_forward()

    # Update the previous plastic strain
    p.vector.axpy(1, dp.vector)  # p = p + 1*du

    # Update the previous stress
    sig_old.x.array[:] = sig.x.array[:]

    # Post-processing
    sig_values = sig.x.array.reshape(ngauss, -1)

    with io.XDMFFile(domain.comm, out_file, "a") as xdmf:
        xdmf.write_function(u, i+1)

    results[i, 0] = new_stretch
    results[i, 1] = np.mean(sig_values[:, 2])
    results[i, 2] = niter

    print(f"---------------------Step {i} done!-----------------------------")

# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time
print(f"Time spent: {elapsed_time} seconds")
# -----------------------------------------------------------------------------------
# PLOT RESULTS
# -----------------------------------------------------------------------------------

plt.figure()
plt.plot(results[:, 0],results[:, 1], "-oC3", label="FEM")
plt.xlabel(r"Strain $\varepsilon_{zz}$")
plt.ylabel(r"Stress $\sigma_{zz}$ [MPa]")
plt.legend()
plt.show()

plt.figure()
plt.bar(np.array(range(0,41)), results[:, 2])
plt.xlabel("Time step")
plt.ylabel("Number of iterations")
plt.show()