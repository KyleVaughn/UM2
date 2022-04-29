using Logging
using MOCNeutronTransport
using Test

# Turn off log info, so we don't clutter up the test output
Logging.disable_logging(Logging.Info)
const Floats = [Float32, Float64, BigFloat]
include("setup/setup_geometry.jl")
include("setup/setup_mesh.jl")
tests = ["quadrature/Quadrature",
         "geometry/Geometry",
         "mesh/Mesh",
        ]
for t in tests
  include("$(t).jl")
end
