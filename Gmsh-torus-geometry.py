#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:13:23 2025

@author: mitja
"""

from mpi4py import MPI

import numpy as np

import dolfinx.plot as plot
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, compute_midpoints, create_unit_cube, create_unit_square, meshtags
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

def create_torus_mesh():
    gmsh.initialize()
    gmsh.model.add("torus")

    R = 1.0  # big radius (center to middle of tube)
    r = 1.0/3.0  # small radius (tube thickness)

    tag = gmsh.model.occ.addTorus(0, 0, 0, R, r)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [tag], 1)

    gmsh.model.mesh.generate(3)
    mesh, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return mesh

msh = create_torus_mesh()
R = 1.0  # big radius (center to middle of tube)
a = 1.0/3.0  # small radius (tube thickness)
plotter = pyvista.Plotter()
plotter.add_text("Torus Mesh and Vector Field", position="upper_edge", font_size=14, color="black")

pyvista_cells, cell_types, x = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, x)

plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")

# Define Nedelec function space and vector field
V = functionspace(msh, ("N1curl", 2))
u = Function(V, dtype=np.float64)
#u.interpolate(lambda x: (np.cos(np.arcsin(x[2]/a))*np.cos(np.arctan(x[1]/x[0])), np.cos(np.arcsin(x[2]/a))*np.sin(np.arctan(x[1]/x[0])), np.sin(np.arcsin(x[2]/a))))
u.interpolate(lambda x: (np.cos(np.arctan(x[2])), np.sin(x[2]), np.zeros(x.shape[1])))

print(u)
# Interpolate to discontinuous Lagrange for visualization
gdim = msh.geometry.dim
V0 = functionspace(msh, ("Discontinuous Lagrange", 2, (gdim,)))
u0 = Function(V0, dtype=np.float64)
u0.interpolate(u)

cells, cell_types, x = plot.vtk_mesh(V0)
grid = pyvista.UnstructuredGrid(cells, cell_types, x)

grid.point_data["u"] = u0.x.array.reshape(x.shape[0], V0.dofmap.index_map_bs)
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
