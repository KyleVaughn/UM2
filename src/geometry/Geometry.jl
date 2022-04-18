module Geometry

using LinearAlgebra
using StaticArrays

import Base: -, +, inv


include("vector.jl")
include("point.jl")
include("plane.jl")
include("axisalignedbox.jl")
include("polytope.jl")
include("interpolate.jl")
#include("geometry/triangulate.jl")
#include("geometry/measure.jl")
end
