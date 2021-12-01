# Routines for extracting segment/face data for tracks (rays) overlaid on a mesh
num_fallback_adjacent = 0
num_fallback_vertices = 0
num_fallback_last_resort = 0

# Return the HRPM/face indices in which each segment resides
# Type-stable, other than warning
function find_segment_faces(segment_points::Vector{Vector{Vector{Point_2D{F}}}},
                            HRPM::HierarchicalRectangularlyPartitionedMesh{F, U},
                            template_vec::MVector{N, U}
                           ) where {F <: AbstractFloat, U <: Unsigned, N}

    @debug "Finding faces corresponding to each segment"
    if !has_materialized_faces(HRPM)
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nγ = length(segment_points)
    bools = fill(false, nγ)
    # Preallocate indices in the most frustrating way
    indices =   [    
                    [ 
                        [ 
                            MVector{N, U}(zeros(U, N)) 
                                for i = 1:length(segment_points[iγ][it])-1 # Segments
                        ] for it = 1:length(segment_points[iγ]) # Tracks
                    ] for iγ = 1:nγ # Angles
                ]
    Threads.@threads for iγ = 1:nγ
        bools[iγ] = find_segment_faces_in_angle!(segment_points[iγ], indices[iγ], HRPM)
    end
    if any(bools)
        iγ_bad = findall(x->!x, bools)
        @error "Failed to find indices for some points in segment_points$iγ_bad"
    end
    return indices
end

# Return the face id in which each segment resides
# Type-stable, other than the warning block
function find_segment_faces(segment_points::Vector{Vector{Vector{Point_2D{F}}}},
                            mesh::UnstructuredMesh_2D{F, U}
                           ) where {F <: AbstractFloat, U <: Unsigned}

    @info "    - Finding faces for each segment"
    if 0 == length(mesh.materialized_faces)
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nγ = length(segment_points)
    bools = fill(false, nγ)
    # Preallocate indices in the most frustrating way
    segment_faces =   [    
                          [ 
                              [ 
                                  U(0) 
                                      for i = 1:length(segment_points[iγ][it])-1 # Segments
                              ] for it = 1:length(segment_points[iγ]) # Tracks
                          ] for iγ = 1:nγ # Angles
                      ]
    Threads.@threads for iγ = 1:nγ
        bools[iγ] = find_segment_faces_in_angle!(segment_points[iγ], segment_faces[iγ], mesh)
    end
    if any(bools)
        iγ_bad = findall(x->!x, bools)
        @error "Failed to find segment faces for some points in angles: $iγ_bad"
    end
    return segment_faces
end

# Ray trace an HRPM given the ray spacing and angular quadrature
# Not type-stable
function ray_trace(tₛ::F,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, F}, 
                   HRPM::HierarchicalRectangularlyPartitionedMesh{F, U}
                   ) where {nᵧ, nₚ, F <: AbstractFloat, U <: Unsigned}
    @info "Ray tracing"
    tracks = generate_tracks(tₛ, ang_quad, HRPM)
    segment_points = segmentize(tracks, HRPM)
    ind_size = node_height(HRPM) + 1
    @info "  - Using the naive segmentize + find face algorithm"
    template_vec = MVector{ind_size, U}(zeros(U, ind_size))
    segment_faces = find_segment_faces(segment_points, HRPM, template_vec)
    return segment_points, segment_faces
end

# Ray trace a mesh given the ray spacing and angular quadrature
# Not type-stable
function ray_trace(tₛ::F,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, F}, 
                   mesh::UnstructuredMesh_2D{F, U}
                   ) where {nᵧ, nₚ, F <: AbstractFloat, U <: Unsigned}
    @info "Ray tracing"
#    global num_fallback_adjacent = 0
#    global num_fallback_vertices = 0
#    global num_fallback_last_resort = 0
    tracks = generate_tracks(tₛ, ang_quad, mesh, boundary_shape = "Rectangle")
    has_mat_edges = 0 < length(mesh.materialized_edges)
    has_mat_faces = 0 < length(mesh.materialized_faces)
    # If the mesh has boundary edges and materialized edges and faces, 
    # use edge-to-edge ray tracing
    if 0 < length(mesh.boundary_edges) && has_mat_edges && has_mat_faces 
        @info "  - Using the edge-to-edge algorithm"
        segment_points, segment_faces = ray_trace_edge_to_edge(tracks, mesh) 
#        @info "    - Adjacent faces fallback   : $num_fallback_adjacent"
#        @info "    - Shared vertices fallback  : $num_fallback_vertices"
#        @info "    - Shared vertices 2 fallback: $num_fallback_last_resort"
        return segment_points, segment_faces
    else
        @info "  - Using the naive segmentize + find face algorithm"
        segment_points = segmentize(tracks, mesh)
        segment_faces = find_segment_faces(segment_points, mesh)
        return segment_points, segment_faces
    end
end

# Get the segment points and the face which the segment lies in for all segments, 
# in all tracks, in all angles, using the edge-to-edge ray tracing method. 
# Assumes a rectangular boundary
function ray_trace_edge_to_edge(tracks::Vector{Vector{LineSegment_2D{F}}},
                                mesh::UnstructuredMesh_2D{F, U}
                                ) where {F <: AbstractFloat, U <: Unsigned}
    # Algorithm info
    alg_str = "    - Using "
    if length(mesh.materialized_edges) != 0
        alg_str = alg_str * "explicit edge intersection, and "
    else
        alg_str = alg_str * "implicit edge intersection, and "
    end
    if length(mesh.materialized_faces) != 0
        alg_str = alg_str * "explicit face intersection in fallback methods"
    else
        alg_str = alg_str * "implicit face intersection in fallback methods"
    end
    @info alg_str

    # Warnings/errors
    if length(mesh.boundary_edges) != 4
        @error "Mesh does not have 4 boundary edges needed for edge-to-edge ray tracing!"
    end

    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(tracks)
    segment_points =[
                        [
                            Point_2D{F}[] for it = 1:length(tracks[iγ]) # Tracks 
                        ] for iγ = 1:nγ # Angles
                    ]
    segment_faces = [
                        [
                            U[] for it = 1:length(tracks[iγ]) # Tracks 
                        ] for iγ = 1:nγ # Angles
                    ]
    # For each angle, get the segments and segment faces for each track
    Threads.@threads for iγ = 1:nγ
        ray_trace_angle_edge_to_edge!(tracks[iγ],
                                      segment_points[iγ],
                                      segment_faces[iγ],
                                      points,
                                      edges,
                                      materialized_edges,
                                      faces,
                                      materialized_faces,
                                      edge_face_connectivity,
                                      face_edge_connectivity,
                                      boundary_edges
                                      )
    end
    return segment_points, segment_faces
end

# Return the points of intersection between the tracks and the mesh edges.
# Returns a Vector{Vector{Vector{Point_2D{F}}}}.
#   index 1 = γ
#   index 2 = track
#   index 3 = point/segment
# Type-stable other than the error message
function segmentize(tracks::Vector{Vector{LineSegment_2D{F}}},
                    HRPM::HierarchicalRectangularlyPartitionedMesh{F, U}
                    ) where {F <: AbstractFloat, U <: Unsigned}

    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(HRPM) 
    @info "Segmentizing using the '$int_alg' algorithm"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(tracks)
    segment_points = Vector{Vector{Vector{Point_2D{F}}}}(undef, nγ)
    Threads.@threads for iγ = 1:nγ
        # for each track, intersect the track with the mesh
        segment_points[iγ] = tracks[iγ] .∩ HRPM
    end
    return segment_points
end

# Return the points of intersection between the tracks and the mesh edges.
# Returns a Vector{Vector{Vector{Point_2D{F}}}}.
#   index 1 = γ
#   index 2 = track
#   index 3 = point/segment
# Type-stable other than the error message
function segmentize(tracks::Vector{Vector{LineSegment_2D{F}}},
                    mesh::UnstructuredMesh_2D{F, U}
                    ) where {F <: AbstractFloat, U <: Unsigned}
    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(mesh) 
    @info "    - Segmentizing using the '$int_alg' algorithm"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(tracks)
    segment_points = Vector{Vector{Vector{Point_2D{F}}}}(undef, nγ)
    Threads.@threads for iγ = 1:nγ
        # for each track, intersect the track with the mesh
        segment_points[iγ] = tracks[iγ] .∩ mesh
    end
    return segment_points
end

# Plot
# -------------------------------------------------------------------------------------------------
# Plot ray tracing data one angle at a time.
function linesegments!(segment_points::Vector{Vector{Vector{Point_2D{T}}}},
                       seg_faces::Vector{Vector{Vector{I}}}) where {T <: AbstractFloat, I <: Unsigned} 
    println("Press enter to plot the segments in the next angle")
    colormap = ColorSchemes.tab20.colors
    lines_by_color = Vector{Vector{LineSegment_2D{T}}}(undef, 20)
    nγ = length(segment_points)
    for iγ = 1:nγ
        for icolor = 1:20
            lines_by_color[icolor] = LineSegment_2D{T}[]
        end
        for it = 1:length(segment_points[iγ])
            for iseg = 1:length(segment_points[iγ][it])-1
                l = LineSegment_2D(segment_points[iγ][it][iseg], segment_points[iγ][it][iseg+1]) 
                face = seg_faces[iγ][it][iseg]
                if face == 0
                    @error "Segment [$iγ][$it][$iseg] has a face id of 0"
                end
                push!(lines_by_color[face % 20 + 1], l)
            end
        end
        for icolor = 1:20
            linesegments!(lines_by_color[icolor], color = colormap[icolor])
        end
        s = readline()
        println(iγ)
    end
end
