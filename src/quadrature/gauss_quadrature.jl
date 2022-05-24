export gauss_quadrature

# Dispatch on Val, so the return Vec length can be inferred.
# Should implement a non-value based version which returns a Vector at some point.
include("legendre_line.jl")      # Accurate to degree 2p-1
include("legendre_hypercube.jl") # Accurate to degree 2p-1
include("legendre_triangle.jl")  # Accurate to degree p
