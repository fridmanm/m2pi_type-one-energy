#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:13:23 2025

@author: mitja
"""

from mpi4py import MPI

import numpy as np

import dolfinx.plot as plot
from dolfinx.fem import Function, functionspace, dirichletbc
from dolfinx.mesh import CellType, compute_midpoints, create_unit_cube, create_unit_square, meshtags,exterior_facet_indices, Mesh, create_mesh, compute_incident_entities
from dolfinx.io import gmshio
import gmsh

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

# If environment variable PYVISTA_OFF_SCREEN is set to true save a png
# otherwise create interactive plot
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)

# Set some global options for all plots
transparent = False
figsize = 800

def create_torus_mesh(resolution):
    gmsh.initialize()
    gmsh.model.add("torus")

    R = 1.0  # big radius (center to middle of tube)
    r = 1.0/3.0  # small radius (tube thickness)

    tag = gmsh.model.occ.addTorus(0, 0, 0, R, r)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize([(0,tag)],resolution)
    gmsh.model.addPhysicalGroup(3, [tag], 1)

    gmsh.model.mesh.generate(3)
    mesh, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return mesh

msh = create_torus_mesh(resolution=0.15)
R = 1.0  # big radius (center to middle of tube)
a = 1.0/3.0  # small radius (tube thickness)
plotter = pyvista.Plotter()
plotter.add_text("Torus Mesh and Vector Field", position="upper_edge", font_size=14, color="black")

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = exterior_facet_indices(msh.topology)
boundary_vertices = compute_incident_entities(msh.topology, boundary_facets, msh.topology.dim-1, 0)

pyvista_cells, cell_types, x = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, x)

plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")


# Define Nedelec function space and vector field
V = functionspace(msh, ("N1curl", 2))
u = Function(V, dtype=np.float64)

def normal_vector(x):
    # Parametric equations for the torus
    theta = np.arctan2(x[1], x[0])
    #if x[0]**2+x[1]**2>=R:
    #    phi = np.arcsin(x[2] / a)
    #else:
    #    phi = np.pi - np.arcsin(x[2] / a)
    #phi=np.arccos((x[0]**2+x[1]**2+x[2]**2+R**2)/(2*R*np.sqrt(x[0]**2+x[1]**2)))
    phi = np.arcsin(x[2] / a)
    phi[np.where(x[0]**2+x[1]**2<R)[0]] = np.pi - np.arcsin(x[2][np.where(x[0]**2+x[1]**2<R)[0]] / a)
    # Normal vector calculation
    nx = np.cos(theta) * np.cos(phi)
    ny = np.sin(theta) * np.cos(phi)
    nz = np.sin(phi)
    
    return (nx, ny, nz)

#u.interpolate(lambda x: (np.cos(np.arcsin(x[2]/a))*np.cos(np.arctan(x[1]/x[0])), np.cos(np.arcsin(x[2]/a))*np.sin(np.arctan(x[1]/x[0])), np.sin(np.arcsin(x[2]/a))))
u.interpolate(lambda x: normal_vector(x))
#print(x)

#surf=exterior_facet_indices(msh.topology.create_connectivity(msh.topology.dim, 0))

# Create a facet mesh to limit the vector field to the surface
'''
entities=np.arange(msh.topology.index_map(2).size_local)

facet_values=np.ones(msh.topology.index_map(2).size_local)

facet_tags = meshtags(msh, 2, entities,facet_values)  # Assuming 2D facets for the surface
V_facet = functionspace(msh, facet_tags)


# Define boundary condition for the function space
def boundary(x):
    return np.full(x.shape[0], True)  # All points are on the boundary

# Apply Dirichlet boundary condition
bc = dirichletbc(u, locate_dofs_geometrically(V, boundary))

# Interpolate to the facet function space for visualization
u_facet = Function(V_facet, dtype=np.float64)
u_facet.interpolate(u)

cells, cell_types, x = plot.vtk_mesh(V_facet)
grid = pyvista.UnstructuredGrid(cells, cell_types, x)

grid.point_data["u"] = u_facet.x.array.reshape(x.shape[0], V_facet.dofmap.index_map_bs)
glyphs = grid.glyph(orient="u", factor=0.1)

plotter.add_mesh(glyphs)
'''

#print(u)
# Interpolate to discontinuous Lagrange for visualization

gdim = msh.geometry.dim
V0 = functionspace(msh, ("Discontinuous Lagrange", 2, (gdim,)))
u0 = Function(V0, dtype=np.float64)
u0.interpolate(u)

cells, cell_types, y = plot.vtk_mesh(V0)
grid = pyvista.UnstructuredGrid(cells, cell_types, y)

grid.point_data["u"] = u0.x.array.reshape(y.shape[0], V0.dofmap.index_map_bs)
glyphs = grid.glyph(orient="u", factor=0.1)

plotter.add_mesh(glyphs)



if pyvista.OFF_SCREEN:
    plotter.screenshot(
        "torus_with_vectors.png",
        transparent_background=transparent,
        window_size=[figsize, figsize],
    )
else:
    plotter.show()
