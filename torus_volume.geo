
//  Parameters of the Torus Volume

R0 = 1.0;     // Major radius (distance from the center to the tube center)
a = 0.3;      // Maximum minor radius (thickness of the tube)
lc = 0.05;    // Characteristic length for mesh


// OpenCASCADE Factory for Advanced Geometries

SetFactory("OpenCASCADE");


// Generate the Torus

Torus(1) = {0, 0, 0, R0, a};


//  Mesh Settings

Mesh.Algorithm = 8;               // Frontal-Delaunay for 3D
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
Mesh 3;


// Save the Mesh

Save "torus_volume.msh";
