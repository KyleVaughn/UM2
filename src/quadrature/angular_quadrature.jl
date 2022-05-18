export ProductAngularQuadrature
export

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
    γ::SVector{nγ,T}    # Azimuthal angles, γ ∈ (0, π)
    wγ::SVector{nγ,T}   # Weights for the azimuthal angles
    θ::SVector{nθ,T}    # Polar angles, θ ∈ (0, π/2)
    wθ::SVector{nθ,T}   # Weights for the polar angles
end

function chebyshev_angular_quadrature(M::Int64)
    # A Chebyshev-type quadrature for a given weight function is a quadrature formula 
    # with equal weights. This function produces evenly spaced angles with equal weights.
    angles = [(π*(2m-1)/(4M)) for m = 1:M]
    weights = zeros(M) .+ 1/M
    return angles, weights
end

# nγ and nθ are azimuthal and polar angles per octant
function generate_angular_quadrature(quadrature_type::String, nγ::Int, nθ::Int)
    if quadrature_type == "Chebyshev-Chebyshev"
        (azi_angles, azi_weights) = generate_chebyshev_angular_quadrature(nγ)
        (pol_angles, pol_weights) = generate_chebyshev_angular_quadrature(nθ)
        append!(azi_angles, reverse(π .- azi_angles))
        azi_weights = azi_weights./2
        append!(azi_weights, azi_weights)
        quadrature = ProductAngularQuadrature(SVector{2nγ}(azi_angles), SVector{2nγ}(azi_weights),
                                              SVector{nθ}(pol_angles), SVector{nθ}(pol_weights))
    else
        @error "Unsupported quadrature type"
    end
    return quadrature
end
