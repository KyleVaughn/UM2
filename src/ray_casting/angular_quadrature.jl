export AbstractQuadratureType, 
       LegendreType, 
       ChebyshevType, 
       ProductAngularQuadrature

export angular_quadrature

abstract type AbstractQuadratureType end
struct ChebyshevType <: AbstractQuadratureType end

# Angular quadrature defined on the unit sphere octant in the upper right, 
# closest to the viewer. The angles and weights are transformed to the other 
# octants by symmetry.
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

struct ProductAngularQuadrature{T <: AbstractFloat}
    wγ::Vector{T}   # Weights for the azimuthal angles
    γ::Vector{T}    # Azimuthal angles, γ ∈ (0, π/2)
    wθ::Vector{T}   # Weights for the polar angles
    θ::Vector{T}    # Polar angles, θ ∈ (0, π/2)
end

function chebyshev_angular_quadrature(M::Integer, ::Type{T}) where {T <: AbstractFloat}
    # A Chebyshev-type quadrature for a given weight function is a quadrature formula 
    # with equal weights. This function produces evenly spaced angles with equal weights.
    weights = fill(T(1) / M, M)
    angles  = Vector{T}(undef, M)
    for m in 1:M
        angles[m] = T(π) * (2 * T(m) - 1) / 4M
    end
    return weights, angles
end

function angular_quadrature(azimuthal_form::AbstractQuadratureType,
                            azimuthal_degree::Integer,
                            polar_form::AbstractQuadratureType,
                            polar_degree::Integer,
                            ::Type{T}) where {T <: AbstractFloat}

    if azimuthal_form isa ChebyshevType
        azi_weights, azi_angles = chebyshev_angular_quadrature(azimuthal_degree, T)
    else
        error("Cannot identify azimuthal quadrature.")
    end

    if polar_form isa ChebyshevType
        pol_weights, pol_angles = chebyshev_angular_quadrature(polar_degree, T)
    else
        error("Cannot identify polar quadrature.")
    end

    return ProductAngularQuadrature{T}(azi_weights, azi_angles, pol_weights, pol_angles)
end
