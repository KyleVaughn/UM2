using Logging
using MOCNeutronTransport
using Test

# Turn off info, so we don't clutter up the test output
Logging.disable_logging(Logging.Info)
const Floats = [Float32, Float64, BigFloat]
const Points1D = [Point1f, Point1, Point1B] 
const Points2D = [Point2f, Point2, Point2B] 
const Points3D = [Point3f, Point3, Point3B] 
include("setup/setup_geometry.jl")
#include("setup/setup_meshes.jl")
tests = ["geometry/Geometry",
#         "mesh/mesh",
        ]
for t in tests
  include("$(t).jl")
end
