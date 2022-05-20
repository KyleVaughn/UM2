export ProductAngularQuadrature
export angular_quadrature

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

# TODO: Function to convert product into general quadrature
#struct GeneralAngularQuadrature{T <: AbstractFloat} <: AngularQuadrature
#    Ω̂::Tuple{Point{T}} # Points on the unit sphere satisfying θ ∈ (0, π/2), γ ∈ (0, π)
#    w::Tuple{T} # Weights for each point
#end

struct ProductAngularQuadrature{nγ, nθ, T}
    wγ::SVector{nγ,T}   # Weights for the azimuthal angles
    γ::SVector{nγ,T}    # Azimuthal angles, γ ∈ (0, π)
    wθ::SVector{nθ,T}   # Weights for the polar angles
    θ::SVector{nθ,T}    # Polar angles, θ ∈ (0, π/2)
end

function angular_quadrature(azimuthal_form::Symbol,
                            azimuthal_degree::Integer,
                            polar_form::Symbol,
                            polar_degree::Integer,
                            type::Type{T}) where {T}
    if azimuthal_form === :chebyshev
        azi_weights_half, azi_angles_half = chebyshev_angular_quadrature(azimuthal_degree, T)
        azi_weights = vcat(azi_weights_half, azi_weights_half)
        azi_angles = vcat(azi_angles_half, reverse(π .- azi_angles_half))
    else
        error("Cannot identify azimuthal quadrature.")
    end
    if polar_form === :chebyshev
        pol_weights, pol_angles = chebyshev_angular_quadrature(polar_degree, T)
    else
        error("Cannot identify azimuthal quadrature.")
    end
    return ProductAngularQuadrature{2azimuthal_degree, polar_degree, T}(
                                       azi_weights, azi_angles,
                                       pol_weights, pol_angles
                                      )
end

function chebyshev_angular_quadrature(M::Integer, type::Type{T}) where {T}
    # A Chebyshev-type quadrature for a given weight function is a quadrature formula 
    # with equal weights. This function produces evenly spaced angles with equal weights.
    weights = SVector(ntuple(m->T(1)/M,           M))
    angles  = SVector(ntuple(m->π*(2T(m) - 1)/4M, M))
    return weights, angles
end
