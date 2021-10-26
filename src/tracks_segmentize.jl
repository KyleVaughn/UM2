# Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html
function tracks(tₛ::T,
                ang_quad::ProductAngularQuadrature{nₐ, nₚ, T},
                HRPM::HierarchicalRectangularlyPartitionedMesh{T}
                ) where {nₐ, nₚ, T <: AbstractFloat}
    w = width(HRPM) 
    h = height(HRPM)
    # Fill the vectors using intersect
    # allocate vectors of cell id for each segment in the track
    # find ireg on each segment
    # return points + cell ids
    #
    nᵧ = length(ang_quad.γ)
    # The tracks for each γ
    tracks = Vector{Vector{LineSegment_2D{T}}}(undef, nᵧ)
    n⃗ₜ    = zeros(Int64, nᵧ)
    γ⃗ₑ    = zeros(T, nᵧ)
    t⃗_eff = zeros(T, nᵧ)
    for (i, γ) in enumerate(ang_quad.γ)
        # Number of tracks in y direction
        n_y = ceil(Int64, w*abs(sin(γ))/tₛ)
        # Number of tracks in x direction
        n_x = ceil(Int64, h*abs(cos(γ))/tₛ)  
        n⃗ₜ[i] = n_y + n_x
        tracks[i] = Vector{LineSegment_2D{T}}(undef, n⃗ₜ[i])

        # Effective angle to ensure cyclic tracks
        γₑ = atan((h*n_x)/(w*n_y))
        if π/2 < γ
            γₑ = γₑ + π/2
        end
        γ⃗ₑ[i] = γₑ

        # Effective ray spacing
        t_eff = w*sin(atan((h*n_x)/(w*n_y)))/n_x
        t⃗_eff[i] = t_eff
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
                tracks[i][ix] = l
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
                tracks[i][n_x + iy] = l
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
                tracks[i][ix] = l
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
                tracks[i][n_y + iy] = l
            end
        end
    end
    return tracks
end

function segmentize(tracks::Vector{Vector{LineSegment_2D{T}}},
                    HRPM::HierarchicalRectangularlyPartitionedMesh{T}
                    ) where {T <: AbstractFloat}

    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(HRPM) 
    @info "Segmentation using $int_alg"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    seg_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, length(tracks))
    seg_cells  = Vector{Vector{Vector{Int64}}}(undef, length(tracks))
    Threads.@threads for iγ = 1:length(tracks)
        # Set up a vector of points for each track
        nt = length(tracks[iγ])
        seg_points[iγ] = Vector{Vector{Point_2D{T}}}(undef, nt)
        seg_cells[iγ] = Vector{Vector{Int64}}(undef, nt)
        # for each track, intersect the track with the mesh
        for it = 1:nt
            seg_points[iγ][it] = tracks[iγ][it] ∩ HRPM
            midpoints = [midpoint(points[i], points[i+1]) for i = 1:length(points)-1]
#            seg_cells[iγ][it] = [
#            for ip = 1:length(seg_points[iγ][it])
#
#            end
        end
    end
    return seg_points
end
