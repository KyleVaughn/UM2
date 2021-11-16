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

# TODO: Function to convert product into general quadrature
#struct GeneralAngularQuadrature{T <: AbstractFloat} <: AngularQuadrature
#    Ω̂::Tuple{Point{T}} # Points on the unit sphere satisfying θ ∈ (0, π/2), γ ∈ (0, π)
#    w::Tuple{T} # Weights for each point
#end

struct ProductAngularQuadrature{M, P, T <: AbstractFloat} <: AngularQuadrature
    γ::NTuple{M, T}    # Azimuthal angles, γ ∈ (0, π)
    w_γ::NTuple{M, T}  # Weights for the azimuthal angles
    θ::NTuple{P, T}    # Polar angles, θ ∈ (0, π/2)
    w_θ::NTuple{P, T}  # Weights for the polar angles
end

function chebyshev_angular_quadrature(M::Int, T::Type{F}) where {F <: AbstractFloat}
    # A Chebyshev-type quadrature for a given weight function is a quadrature formula with equal
    # weights. This function produces evenly spaced angles with equal weights.
    angles = T[(π*(2m-1)/(4M)) for m = M:-1:1]
    weights = zeros(T, M) .+ T(1/M)
    return angles, weights
end

# nγ and nθ are azimuthal and polar angles per octant
function angular_quadrature(quadrature_type::String, nγ::Int, nθ::Int;
                            T::Type{F}=Float64) where {F <: AbstractFloat}
    if quadrature_type == "Chebyshev-Chebyshev"
        (azi_angles, azi_weights) = chebyshev_angular_quadrature(nγ, T)
        (pol_angles, pol_weights) = chebyshev_angular_quadrature(nθ, T)
        append!(azi_angles, [π - azi_angles[i] for i = 1:nγ])
        azi_weights = azi_weights./2
        append!(azi_weights, azi_weights)
        quadrature = ProductAngularQuadrature(Tuple(azi_angles), Tuple(azi_weights),
                                              Tuple(pol_angles), Tuple(pol_weights))
    else
        @error "Unsupported quadrature type"
    end
    return quadrature
end
