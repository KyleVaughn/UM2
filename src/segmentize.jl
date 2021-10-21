# Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html
function segmentize(tₛ::T,
                    ang_quad::ProductAngularQuadrature{nₐ, nₚ, T},
                    HRPM::HierarchicalRectangularlyPartitionedMesh{T}
                    ) where {nₐ, nₚ, T <: AbstractFloat}
    w = width(HRPM) 
    h = height(HRPM)
    # Iterate over the angles such that the effective angle sweeps from 0 to π
    n_γ = length(ang_quad.γ) ÷ 2
    azi_iterator = [i for i in n_γ:-1:1]
    append!(azi_iterator, [i for i in 2*n_γ:-1:n_γ+1])
    for i = azi_iterator 
        γ = ang_quad.γ[i]
        # Number of tracks in y direction
        n_y = floor(w/tₛ*abs(sin(γ))) + 1
        # Number of tracks in x direction
        n_x = floor(h/tₛ*abs(cos(γ))) + 1   
        # Effective angle to ensure cyclic tracks
        γ_e = atan((h*n_x)/(w*n_y))
        if π/2 < γ
            γ_e = γ_e + π/2
        end
        # Effective ray spacing
        t_eff = w*sin(atan((h*n_x)/(w*n_y)))/n_x

        # Generate tracks
        # Give info about which algorithm is being used
        # face, edge, implicit, explicit
    end
end
