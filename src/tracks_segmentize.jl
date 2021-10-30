# Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html
function tracks(tₛ::T,
                ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T},
                HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    w = width(HRPM) 
    h = height(HRPM)
    # The tracks for each γ
    the_tracks = [ tracks(tₛ, w, h, γ) for γ in ang_quad.γ ]  
    # Shift all tracks if necessary, since the tracks are generated as if the HRPM has a 
    # bottom left corner at (0,0)
    offset = HRPM.rect.points[1]
    for angle in the_tracks
        for track in angle
            track = LineSegment_2D(track.points[1] + offset, track.points[2] + offset)
        end
    end
    return the_tracks
end

function tracks(tₛ::T, w::T, h::T, γ::T) where {T <: AbstractFloat}
    # Number of tracks in y direction
    n_y = ceil(Int64, w*abs(sin(γ))/tₛ)
    # Number of tracks in x direction
    n_x = ceil(Int64, h*abs(cos(γ))/tₛ)  
    # Total number of tracks
    nₜ = n_y + n_x
    # Allocate the tracks
    the_tracks = Vector{LineSegment_2D{T}}(undef, nₜ)
    # Effective angle to ensure cyclic tracks
    γₑ = atan((h*n_x)/(w*n_y))
    if π/2 < γ
        γₑ = γₑ + T(π/2)
    end
    # Effective ray spacing for the cyclic tracks
    t_eff = w*sin(atan((h*n_x)/(w*n_y)))/n_x
    if γₑ ≤ π/2
        # Generate tracks from the bottom edge of the rectangular domain
        for ix = 1:n_x
            x₀ = w - t_eff*T(ix - 0.5)/sin(γₑ)
            y₀ = T(0)
            # Segment either terminates at the right edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = min(w, h/tan(γₑ) + x₀)
            y₁ = min((w - x₀)*tan(γₑ), h) 
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arc_length(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            the_tracks[ix] = l
        end
        # Generate tracks from the left edge of the rectangular domain
        for iy = 1:n_y
            x₀ = T(0) 
            y₀ = t_eff*T(iy - 0.5)/cos(γₑ)
            # Segment either terminates at the right edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = min(w, (h - y₀)/tan(γₑ))
            y₁ = min(w*tan(γₑ) + y₀, h)
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arc_length(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            the_tracks[n_x + iy] = l
        end
    else
        # Generate tracks from the bottom edge of the rectangular domain
        for ix = n_y:-1:1
            x₀ = w - t_eff*T(ix - 0.5)/sin(γₑ)
            y₀ = T(0)
            # Segment either terminates at the left edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = max(0, h/tan(γₑ) + x₀)
            y₁ = min(x₀*abs(tan(γₑ)), h) 
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arc_length(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            the_tracks[ix] = l
        end
        # Generate tracks from the right edge of the rectangular domain
        for iy = 1:n_x
            x₀ = w
            y₀ = t_eff*T(iy - 0.5)/abs(cos(γₑ))
            # Segment either terminates at the left edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = max(0, w + (h - y₀)/tan(γₑ))
            y₁ = min(w*abs(tan(γₑ)) + y₀, h)
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arc_length(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            the_tracks[n_y + iy] = l
        end
    end
    return the_tracks
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
    nlevels = levels(HRPM)
    seg_faces  = Vector{Vector{Vector{MVector{nlevels, Int64}}}}(undef, length(tracks))
    Threads.@threads for iγ = 1:length(tracks)
        # Set up a vector of points for each track
        nt = length(tracks[iγ])
        seg_points[iγ] = Vector{Vector{Point_2D{T}}}(undef, nt)
        seg_faces[iγ] = Vector{Vector{Int64}}(undef, nt)
        # for each track, intersect the track with the mesh
        for it = 1:nt
            seg_points[iγ][it] = tracks[iγ][it] ∩ HRPM
            npoints = length(seg_points[iγ][it])
            seg_faces[iγ][it] = [MVector{nlevels, Int64}(zeros(Int64, nlevels)) for i = 1:npoints - 1] 
            for iseg = 1:npoints-1
                p_midpoint = midpoint(seg_points[iγ][it][iseg], seg_points[iγ][it][iseg+1])
                find_face(p_midpoint, HRPM, seg_faces[iγ][it][iseg])
            end
        end
    end
    return (seg_points, seg_faces)
end

function segmentize_no_face_data(tracks::Vector{Vector{LineSegment_2D{T}}},
                    HRPM::HierarchicalRectangularlyPartitionedMesh{T}
                    ) where {T <: AbstractFloat}

    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(HRPM) 
    @info "Segmentation using $int_alg"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    seg_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, length(tracks))
    Threads.@threads for iγ = 1:length(tracks)
        # Set up a vector of points for each track
        nt = length(tracks[iγ])
        seg_points[iγ] = Vector{Vector{Point_2D{T}}}(undef, nt)
        # for each track, intersect the track with the mesh
        for it = 1:nt
            seg_points[iγ][it] = tracks[iγ][it] ∩ HRPM
            npoints = length(seg_points[iγ][it])
            for iseg = 1:npoints-1
                p_midpoint = midpoint(seg_points[iγ][it][iseg], seg_points[iγ][it][iseg+1])
            end
        end
    end
    return seg_points
end

function face_data(seg_points::Vector{Vector{Vector{Point_2D{T}}}},
                   HRPM::HierarchicalRectangularlyPartitionedMesh{T}
                    ) where {T <: AbstractFloat}

    println("Running face_data")
    nlevels = levels(HRPM)
    seg_faces  = Vector{Vector{Vector{MVector{nlevels, Int64}}}}(undef, length(seg_points))
    Threads.@threads for iγ = 1:length(seg_points)
        nt = length(seg_points[iγ])
        seg_faces[iγ] = Vector{Vector{Int64}}(undef, nt)
        # for each track, intersect the track with the mesh
        for it = 1:nt
            npoints = length(seg_points[iγ][it])
            seg_faces[iγ][it] = [MVector{nlevels, Int64}(zeros(Int64, nlevels)) for i = 1:npoints - 1] 
            for iseg = 1:npoints-1
                p_midpoint = midpoint(seg_points[iγ][it][iseg], seg_points[iγ][it][iseg+1])
                find_face(p_midpoint, HRPM, seg_faces[iγ][it][iseg])
            end
        end
    end
    return seg_faces
end
