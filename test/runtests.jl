using MOCNeutronTransport
using Test
using Logging

# TODO: centroid.jl, boundingbox.jl, cartesian_to_parametric.jl

Logging.disable_logging(Logging.Info)
include("setup/setup_primitives.jl")
include("setup/setup_meshes.jl")
tests = ["SVector",
         "primitives/Point",
         "primitives/LineSegment",
         "primitives/QuadraticSegment",
         "primitives/Hyperplane",
         "primitives/AABox",
         "primitives/ConvexPolygon",
         "primitives/QuadraticPolygon",
         "primitives/ConvexPolyhedron",
         "primitives/QuadraticPolyhedron",
         "mesh/mesh_IO",
         "interpolation",
         "jacobian",
         "triangulate",
         "measure",
#        # "./mesh/UnstructuredMesh_2D",
         #"./mesh/IO_abaqus",
#         "./mesh/IO_vtk",
#         "./mesh/IO_xdmf",
#         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
