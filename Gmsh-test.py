#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 11:37:42 2025

@author: mitja
"""

from mpi4py import MPI
from dolfin import *
#from dolfinx.io import XDMFFile, gmshio

try:
    import gmsh  # type: ignore
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)
    
def gmsh_sphere(model: gmsh.model, name: str) -> gmsh.model:
    """Create a Gmsh model of a sphere.

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a sphere mesh added.

    """
    model.add(name)
    model.setCurrent(name)
    sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical marker for cells. It is important to call this
    # function after OpenCascade synchronization
    model.add_physical_group(dim=3, tags=[sphere])

    # Generate the mesh
    model.mesh.generate(dim=3)
    return model


def gmsh_sphere_minus_box(model: gmsh.model, name: str) -> gmsh.model:
    """Create a Gmsh model of a sphere with a box from the sphere removed.

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a sphere mesh added.
    """
    model.add(name)
    model.setCurrent(name)

    sphere_dim_tags = model.occ.addSphere(0, 0, 0, 1)
    box_dim_tags = model.occ.addBox(0, 0, 0, 1, 1, 1)
    model_dim_tags = model.occ.cut([(3, sphere_dim_tags)], [(3, box_dim_tags)])
    model.occ.synchronize()

    # Add physical tag 1 for exterior surfaces
    boundary = model.getBoundary(model_dim_tags[0], oriented=False)
    boundary_ids = [b[1] for b in boundary]
    model.addPhysicalGroup(2, boundary_ids, tag=1)
    model.setPhysicalName(2, 1, "Sphere surface")

    # Add physical tag 2 for the volume
    volume_entities = [model[1] for model in model.getEntities(3)]
    model.addPhysicalGroup(3, volume_entities, tag=2)
    model.setPhysicalName(3, 2, "Sphere volume")

    model.mesh.generate(dim=3)
    return model


def gmsh_ring(model: gmsh.model, name: str) -> gmsh.model:
    """Create a Gmsh model of a ring-type geometry using hexahedral cells.

    Args:
        model: Gmsh model to add the mesh to.
        name: Name (identifier) of the mesh to add.

    Returns:
        Gmsh model with a sphere mesh added.
    """
    model.add(name)
    model.setCurrent(name)

    # Recombine tetrahedra to hexahedra
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    circle = model.occ.addDisk(0, 0, 0, 1, 1)
    circle_inner = model.occ.addDisk(0, 0, 0, 0.5, 0.5)
    cut = model.occ.cut([(2, circle)], [(2, circle_inner)])[0]
    extruded_geometry = model.occ.extrude(cut, 0, 0, 0.5, numElements=[5], recombine=True)
    model.occ.synchronize()

    model.addPhysicalGroup(2, [cut[0][1]], tag=1)
    model.setPhysicalName(2, 1, "2D cylinder")
    boundary_entities = model.getEntities(2)
    other_boundary_entities = []
    for entity in boundary_entities:
        if entity != cut[0][1]:
            other_boundary_entities.append(entity[1])
    model.addPhysicalGroup(2, other_boundary_entities, tag=3)
    model.setPhysicalName(2, 3, "Remaining boundaries")

    model.mesh.generate(3)
    model.mesh.setOrder(2)
    volume_entities = []
    for entity in extruded_geometry:
        if entity[0] == 3:
            volume_entities.append(entity[1])
    model.addPhysicalGroup(3, volume_entities, tag=1)
    model.setPhysicalName(3, 1, "Mesh volume")
    return model

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

# Create model
model = gmsh.model()

model = gmsh_sphere(model, "Sphere")
model.setCurrent("Sphere")
create_mesh(MPI.COMM_SELF, model, "sphere", f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w")

model = gmsh_sphere_minus_box(model, "Sphere minus box")
model.setCurrent("Sphere minus box")
create_mesh(MPI.COMM_WORLD, model, "ball_d1", "out_gmsh/mesh.xdmf", "w")

model.mesh.generate(3)
gmsh.option.setNumber("General.Terminal", 1)
model.mesh.setOrder(2)
gmsh.option.setNumber("General.Terminal", 0)
create_mesh(MPI.COMM_WORLD, model, "ball_d2", "out_gmsh/mesh.xdmf", "a")

model = gmsh_ring(model, "Hexahedral mesh")
model.setCurrent("Hexahedral mesh")
create_mesh(MPI.COMM_WORLD, model, "hex_d2", "out_gmsh/mesh.xdmf", "a")