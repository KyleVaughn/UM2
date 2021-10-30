function ray_trace(tₛ::T,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T}, 
                   HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                   ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    the_tracks = tracks(tₛ, ang_quad, HRPM)
    segment_points = segmentize(the_tracks, HRPM)
    face_indices = segment_face_indices(segment_points, HRPM)
    return segment_points, face_indices
end

function segmentize(tracks::Vector{Vector{LineSegment_2D{T}}},
                    HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                    ) where {T <: AbstractFloat, I <: Unsigned}

    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(HRPM) 
    @info "Segmentizing using the '$int_alg' algorithm"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    seg_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, length(tracks))
    Threads.@threads for iγ = 1:length(tracks)
        # for each track, intersect the track with the mesh
        seg_points[iγ] = tracks[iγ] .∩ HRPM
    end
    return seg_points
end

function segment_face_indices(seg_points::Vector{Vector{Vector{Point_2D{T}}}},
                              HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                             ) where {T <: AbstractFloat, I <: Unsigned}

    @info "Finding face indices corresponding to each segment"
    if !are_faces_materialized(HRPM)
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nlevels = levels(HRPM)
    nγ = length(seg_points)
    # Preallocate indices in the most frustrating way
    indices =   [    
                    [ 
                        [ 
                            MVector{nlevels, I}(zeros(I, nlevels)) 
                                for i = 1:length(seg_points[iγ][it])-1 # Segments
                        ] for it = 1:length(seg_points[iγ]) # Tracks
                    ] for iγ = 1:nγ # Angles
                ]
    Threads.@threads for iγ = 1:nγ
        segment_face_indices(iγ, seg_points[iγ], indices[iγ], HRPM)
    end
    return indices
end

# Get the face indices for all segments in a single track
function segment_face_indices(points::Vector{Point_2D{T}},
                              indices::Vector{MVector{N, I}},
                              HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                             ) where {T <: AbstractFloat, I <: Unsigned, N}
    # Points in the track
    npoints = length(points)
    bools = fill(false, npoints-1)
    # Test the midpoint of each segment to find the face
    for iseg = 1:npoints-1
        p_midpoint = midpoint(points[iseg], points[iseg+1])
        bools[iseg] = find_face(p_midpoint, indices[iseg], HRPM)
    end
    return all(bools)
end

# Get the face indices for all tracks in a single angle
function segment_face_indices(iγ::Int64,
                              points::Vector{Vector{Point_2D{T}}},
                              indices::Vector{Vector{MVector{N, I}}},
                              HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                             ) where {T <: AbstractFloat, I <: Unsigned, N}
    nt = length(points)
    bools = fill(false, nt)
    # for each track, find the segment indices
    for it = 1:nt
        # Points in the track
        npoints = length(points[it])
        # Returns true if indices were found for all segments in the track
        bools[it] = segment_face_indices(points[it], indices[it], HRPM)
    end
    if !all(bools)
        it_bad = findall(x->!x, bools)
        @warn "Failed to find indices for some points in seg_points[$iγ][$it_bad]"
    end
    return all(bools)
end

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
