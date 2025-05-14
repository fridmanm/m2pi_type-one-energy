from dolfin import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D mesh
mesh = UnitCubeMesh(10, 10, 10)

# Function space for scalar ψ
V = FunctionSpace(mesh, "Lagrange", 1)

# Dirichlet boundary condition for ψ
psi_bc_expr = Expression("1 + x[0]*x[0] + 2*x[1]*x[1] + 3*x[2]*x[2]", degree=2)
bc = DirichletBC(V, psi_bc_expr, "on_boundary")

# Define the Helmholtz parameter μ
mu = 2.0
psi = TrialFunction(V)
v = TestFunction(V)

# Helmholtz variational formulation
a = dot(grad(psi), grad(v))*dx - mu*2 * psi*v*dx
L = Constant(0)*v*dx  # RHS = 0

# Solve for ψ
psi_sol = Function(V)
solve(a == L, psi_sol, bc)

# Compute gradient of ψ
V_vec = VectorFunctionSpace(mesh, "CG", 1)
grad_psi = project(grad(psi_sol), V_vec)

# Compute r × ∇ψ
x, y, z = SpatialCoordinate(mesh)
rxgradpsi = as_vector([
    y*grad_psi[2] - z*grad_psi[1],
    z*grad_psi[0] - x*grad_psi[2],
    x*grad_psi[1] - y*grad_psi[0]
])
B = project(rxgradpsi, V_vec)
plot(B)
#plot(mesh)
plt.show()


# Extract values from the scalar solution
coords = mesh.coordinates()
values = psi_sol.compute_vertex_values(mesh)

# Create 3D scatter plot of ψ
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap='viridis', s=10)
plt.colorbar(sc, label="ψ value")
ax.set_title("Scalar Field ψ in 3D")	
plt.show()

# Save the vector field B
#File("B_vector.pvd") << B