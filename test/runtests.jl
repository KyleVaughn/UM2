using Logging
using MOCNeutronTransport
using StaticArrays
using Test

# Turn of info, so we don't clutter up the test output
Logging.disable_logging(Logging.Info)
const Floats = [Float16, Float32, Float64, BigFloat]
#include("setup/setup_primitives.jl")
#include("setup/setup_meshes.jl")
tests = ["SVector",
         "primitives/primitives",
#         "mesh/mesh",
#         "mesh/PolygonMesh",
#         "mesh/mesh_IO",
#         "interpolation",
#         "jacobian",
#         "triangulate",
#         "measure",
#        # "./mesh/UnstructuredMesh_2D",
         #"./mesh/IO_abaqus",
#         "./mesh/IO_vtk",
#         "./mesh/IO_xdmf",
#         "AngularQuadrature", 
        ]
for t in tests
  include("$(t).jl")
end
