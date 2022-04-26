export AbstractRefShape, RefSimplex, RefHypercube, RefLine, RefTriangle, RefTetrahedron,
       RefSquare, RefCube 
export gauss_quadrature

abstract type AbstractRefShape end
struct RefSimplex{K} <: AbstractRefShape end
struct RefHypercube{K} <: AbstractRefShape end

const RefTriangle    = RefSimplex{2}
const RefTetrahedron = RefSimplex{3}

const RefLine        = RefHypercube{1}
const RefSquare      = RefHypercube{2}
const RefCube        = RefHypercube{3}

# Dispatch on Val, so the return Vec length can be inferred.
# Should implement a non-value based version which returns a Vector at some point.
include("legendre_RefLine.jl")      # Accurate to degree 2p-1
include("legendre_RefHyperCube.jl") # Accurate to degree 2p-1
include("legendre_RefTriangle.jl")  # Accurate to degree p
