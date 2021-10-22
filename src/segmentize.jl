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
    for iγ = azi_iterator 
        # TODO: Account for γ_e > π/2
        f = Figure()
        ax = Axis(f[1, 1], aspect = 1)
        linesegments!(HRPM.rect)
        display(f)
        γ = ang_quad.γ[iγ]
        # Number of tracks in y direction
        n_y = floor(Int64, w/tₛ*abs(sin(γ))) + 1
        # Number of tracks in x direction
        n_x = floor(Int64, h/tₛ*abs(cos(γ))) + 1   
        # Effective angle to ensure cyclic tracks
        γ_e = atan((h*n_x)/(w*n_y))
        if π/2 < γ
            γ_e = γ_e + π/2
        end
        # Effective ray spacing
        t_eff = w*sin(atan((h*n_x)/(w*n_y)))/n_x
        # Generate tracks
        for ix = 1:n_x
            x₀ = w - t_eff*ix 
            y₀ = T(0)
            # Segment either terminates at the right edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = min(w, h/tan(γ_e) + x₀)
            y₁ = min(t_eff*ix*tan(γ_e), h) 
            linesegments!(LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁)))
            println(x₁, " ", y₁)
            s = readline()
        end
        for iy = 1:n_y
            x₀ = T(0) 
            y₀ = t_eff*iy
            # Segment either terminates at the right edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = min(w, (h - y₀)/tan(γ_e))
            y₁ = min(w*tan(γ_e) + y₀, h)
            linesegments!(LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁)))
            println(x₁, " ", y₁)
            s = readline()
        end
        # Give info about which algorithm is being used
        # face, edge, implicit, explicit
    end
end
