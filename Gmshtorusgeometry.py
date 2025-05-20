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
from dolfinx.mesh import exterior_facet_indices, compute_incident_entities
from dolfinx.io import gmshio
import gmsh
import pyvista
from dolfinx.mesh import create_submesh

# Initialize pyvista
if pyvista.OFF_SCREEN:
    pyvista.start_xvfb(wait=0.1)

def create_torus_mesh(resolution):
    gmsh.initialize()
    gmsh.model.add("torus")

    R = 1.0  # big radius (center to middle of tube)
    r = 1.0/3.0  # small radius (tube thickness)

    tag = gmsh.model.occ.addTorus(0, 0, 0, R, r)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize([(0,tag)], resolution)
    gmsh.model.addPhysicalGroup(3, [tag], 1)

    gmsh.model.mesh.generate(3)
    mesh, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return mesh




# Define normal vector function
def normal_vector(x,R,a):
    theta = np.arctan2(x[1], x[0])
    phi = np.arcsin(x[2] / a)
    
    # For points inside the torus (inner surface)
    inner_mask = (x[0]**2 + x[1]**2) < R
    phi[inner_mask] = np.pi - np.arcsin(x[2][inner_mask] / a)
    
    # Normal vector calculation
    nx = np.cos(theta) * np.cos(phi)
    ny = np.sin(theta) * np.cos(phi)
    nz = np.sin(phi)
    
    # Normalize the vectors
    #norm = np.sqrt(nx**2 + ny**2 + nz**2)
    #nx /= norm
    #ny /= norm
    #nz /= norm
    
    return (nx, ny, nz)

def normal_field(resolution=0.15,R=1.0,a=1.0/3.0, visual=False):    
    # Create mesh
    msh = create_torus_mesh(resolution=0.15)

    # Create plotter
    plotter = pyvista.Plotter()
    plotter.add_text("Torus Mesh and Vector Field", position="upper_edge", font_size=14, color="black")
    # Get boundary facets and vertices
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = exterior_facet_indices(msh.topology)
    #boundary_vertices = compute_incident_entities(msh.topology, boundary_facets, fdim, 0)

    # Create a function space on the boundary only
    # First, we need to create a boundary mesh

    boundary_mesh, entity_map = create_submesh(msh, fdim, boundary_facets)[:2]

    # Now define a function space on the boundary mesh
    V_boundary = functionspace(boundary_mesh, ("Lagrange", 1, (boundary_mesh.geometry.dim,)))
    u_boundary = Function(V_boundary, dtype=np.float64)

    pyvista_cells, cell_types, x = plot.vtk_mesh(boundary_mesh)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, x)
    plotter.add_mesh(grid, style="wireframe", line_width=2, color="black")

    # Interpolate the normal vector function on the boundary
    u_boundary.interpolate(lambda x: normal_vector(x,R,a))

    # Create mesh for visualization

    gdim = boundary_mesh.geometry.dim
    V0 = functionspace(boundary_mesh, ("Discontinuous Lagrange", 2, (gdim,)))
    u0 = Function(V0, dtype=np.float64)
    u0.interpolate(u_boundary)
    
    if visual:
        # Visualize the boundary mesh and vector field
        cells, cell_types, y = plot.vtk_mesh(V0)
        boundary_grid = pyvista.UnstructuredGrid(cells, cell_types, y)

        # Get the vector field data
        boundary_grid.point_data["u"] = u0.x.array.reshape(y.shape[0], 3)

        # Create glyphs for visualization
        glyphs = boundary_grid.glyph(orient="u", factor=0.1)

        plotter.add_mesh(glyphs, color="red")

        if pyvista.OFF_SCREEN:
            plotter.screenshot(
                "torus_with_vectors.png",
                transparent_background=False,
                window_size=[800, 800],
                )
        else:
            plotter.show()
            
    return u0

#n=normal_field(True)