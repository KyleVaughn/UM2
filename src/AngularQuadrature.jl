# Angular quadrature defined on the unit sphere octant in the upper right, closest to the viewer.
# The angles and weights are transformed to the other octants by symmetry.
#     +----+----+
#    /    /    /|
#   +----+----+ |
#  /    /    /| + 
# +----+----+ |/|
# |    |this| + |
# |    | one|/| +
# +----+----+ |/
# |    |    | +
# |    |    |/
# +----+----+
#
# The spherical coordinates are defined in the following manner 
# Ω̂ = (Ω_i, Ω_j, Ω_k) = (cos(θ),  sin(θ)cos(γ),  sin(θ)sin(γ))
#                     = (     μ, √(1-μ²)cos(γ), √(1-μ²)sin(γ)) 
#
#        j
#        ^
#        |   θ is the polar angle about the i-axis (x-direction)
#        |   γ is the azimuthal angle in the j-k plane, from the j-axis 
#        |
#        |
#        |
#       /|
#      (γ|
#       \|--------------------> i
#       / \θ)
#      /   \
#     /     \
#    /       \ Ω̂
#   /         v
#  𝘷
#  k
abstract type AngularQuadrature end

struct GeneralAngularQuadrature{T <: AbstractFloat} <: AngularQuadrature
    Ω̂::Vector{Point{T}} # Points on the unit sphere satisfying θ ∈ (0, π/2), γ ∈ (0, π/2)
    w::Vector{T} # Weights for each point
end

struct ProductAngularQuadrature{T <: AbstractFloat} <: AngularQuadrature
    θ::Vector{T}    # Polar angles, θ ∈ (0, π/2)
    γ::Vector{T}    # Azimuthal angles, γ ∈ (0, π/2)
    w_θ::Vector{T}  # Weights for the polar angles
    w_γ::Vector{T}  # Weishts for the azimuthal angles    
end

ProductAngularQuadrature((θ, w_θ), (γ,w_γ)) = ProductAngularQuadrature(θ, γ, w_θ, w_γ)


function chebyshev_angular_quadrature(M::Int)
    # A Chebyshev-type quadrature for a given weight function is a quadrature formula with equal 
    # weights. This function produces evenly spaced angles with equal weights.
    angles = [π*(2m-1)/4M for m = 1:M]
    weights = [1.0/M for m = 1:M]
    return angles, weights
end

function AngularQuadrature(quadrature_type::String, M::Int, N::Int)
    if quadrature_type == "Chebyshev-Chebyshev"
        quadrature = ProductAngularQuadrature(chebyshev_angular_quadrature(M), 
                                              chebyshev_angular_quadrature(N))
    else
        ArgumentError("Unsupported quadrature type.")
    end
    return quadrature
end

# TODO: Function to convert product into general quadrature
