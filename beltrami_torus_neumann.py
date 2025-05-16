#previous part of the code is the same, I just added boundry conditions around line 25 onwward, not sure if it works after discussing with other team members


from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import fem
from ufl import TrialFunction, TestFunction, curl, inner, cross, dx, ds, FacetNormal, dot
from petsc4py import PETSc
import numpy as np
from dolfinx import log, default_scalar_type
from dolfinx import fem, mesh, plot

# 1. Read torus mesh from Gmsh 
domain, cell_tags, facet_tags = gmshio.read_from_msh("torus.msh", MPI.COMM_WORLD, gdim=3)

# 2. Define function space
V = fem.functionspace(domain, ("N1curl",1, (domain.topology.dim,)))
u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(domain)
mu = 1

a = inner(u, curl(v)) * dx + inner(n, cross(u, v)) * ds - mu * inner(u, v) * dx 

f = fem.Constant(domain, PETSc.ScalarType((1,0,0)))

T = fem.Constant(domain, default_scalar_type((0, 0, 0)))


tdim = domain.topology.dim
facets = mesh.exterior_facet_indices(domain.topology)
neumann_tag = 75
facet_markers = mesh.meshtags(domain, tdim-1, facets, neumann_tag)










#neumann boundry condition as weak form?
#try dot after ds
L1 = inner(f, v) * dx + dot(v,n) * ds(neumann_tag)


# Find a point to fix (arbitrary choice)




#L1 = inner(f, v) * dx + inner(v,n) * ds

# Left boundary (x=0)


# Combine boundary conditions



#3. Define Boundry conditions



from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L1, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Interpolate for visualization (Lagrange)
V_cg = fem.functionspace(domain, ("Lagrange", 1, (domain.topology.dim,)))
uh_cg = fem.Function(V_cg)
uh_cg.interpolate(uh)

from dolfinx.io import XDMFFile
with XDMFFile(MPI.COMM_WORLD, "torus_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh_cg)

