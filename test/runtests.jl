using Logging
using Test
using UM2 

# Turn off log info, so we don't clutter up the test output
Logging.disable_logging(Logging.Info)
const Floats = [Float32, Float64, BigFloat]
# include("setup/setup_geometry.jl")
# include("setup/setup_mesh.jl")
tests = ["common/Common",
         "quadrature/Quadrature",
#         "geometry/Geometry",
#         "mesh/Mesh",
#         "raytracing/Raytracing"
        ]
for t in tests
  include("$(t).jl")
end
