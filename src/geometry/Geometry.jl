module Geometry

using LinearAlgebra
using StaticArrays

import Base: -, +, inv, zero

include("vector.jl")
include("point.jl")
include("plane.jl")
include("axisalignedbox.jl")
include("polytope.jl")
include("polytopes/interpolate.jl")
include("polytopes/polynomial.jl")
include("polytopes/quadraticsegment.jl")
include("triangulate.jl")
include("measure.jl")
end
