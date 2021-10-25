# Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html
function segmentize(tₛ::T,
                    ang_quad::ProductAngularQuadrature{nₐ, nₚ, T},
                    HRPM::HierarchicalRectangularlyPartitionedMesh{T}
                    ) where {nₐ, nₚ, T <: AbstractFloat}
    w = width(HRPM) 
    h = height(HRPM)
    # Loop through all angles to determine the effective angle and number of tracks for 
    # each azimuthal angle.
    # Allocate vectors of points for each direction
    # Generate the line across the geom,
    # Fill the vectors using intersect
    # allocate vectors of cell id for each segment in the track
    # find ireg on each segment
    # return points + cell ids
    for γ in ang_quad.γ
        # Number of tracks in y direction
        n_y = ceil(Int64, w*abs(sin(γ))/tₛ)
        # Number of tracks in x direction
        n_x = ceil(Int64, h*abs(cos(γ))/tₛ)  
        # Effective angle to ensure cyclic tracks
        γₑ = atan((h*n_x)/(w*n_y))
        if π/2 < γ
            γₑ = γₑ + π/2
        end
        # Effective ray spacing
        t_eff = w*sin(atan((h*n_x)/(w*n_y)))/n_x
        if γₑ ≤ π/2
            # Generate tracks
            for ix = 1:n_x
                x₀ = w - t_eff*(ix - 0.5)/sin(γₑ)
                y₀ = T(0)
                # Segment either terminates at the right edge of the rectangle
                # Or on the top edge of the rectangle
                x₁ = min(w, h/tan(γₑ) + x₀)
                y₁ = min((w - x₀)*tan(γₑ), h) 
                l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
                if arc_length(l) < 1e-6
                    @warn "Small segment for $l"
                end
                println("($x₀, $y₀) ($x₁, $y₁)")
                s = readline()
            end
            for iy = 1:n_y
                x₀ = T(0) 
                y₀ = t_eff*(iy - 0.5)/cos(γₑ)
                # Segment either terminates at the right edge of the rectangle
                # Or on the top edge of the rectangle
                x₁ = min(w, (h - y₀)/tan(γₑ))
                y₁ = min(w*tan(γₑ) + y₀, h)
                l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
                if arc_length(l) < 1e-6
                    @warn "Small segment for $l"
                end
                s = readline()
            end
        else
            # Generate tracks
            for ix = n_y:-1:1
                x₀ = w - t_eff*(ix - 0.5)/sin(γₑ)
                y₀ = T(0)
                # Segment either terminates at the left edge of the rectangle
                # Or on the top edge of the rectangle
                x₁ = max(0, h/tan(γₑ) + x₀)
                y₁ = min(x₀*abs(tan(γₑ)), h) 
                l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
                if arc_length(l) < 1e-6
                    @warn "Small segment for $l"
                end
            end
            for iy = 1:n_x
                x₀ = w
                y₀ = t_eff*(iy - 0.5)/abs(cos(γₑ))
                # Segment either terminates at the left edge of the rectangle
                # Or on the top edge of the rectangle
                x₁ = max(0, w + (h - y₀)/tan(γₑ))
                y₁ = min(w*abs(tan(γₑ)) + y₀, h)
                l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
                if arc_length(l) < 1e-6
                    @warn "Small segment for $l"
                end
            end
        end
        # Give info about which algorithm is being used
        # face, edge, implicit, explicit
    end
end
