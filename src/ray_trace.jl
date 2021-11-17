# Routines for extracting segment/face data for tracks (rays) overlaid on a mesh

# Return the HRPM/face indices in which each segment resides
function find_segment_faces(segment_points::Vector{Vector{Vector{Point_2D{T}}}},
                            HRPM::HierarchicalRectangularlyPartitionedMesh{T, I},
                            template_vec::MVector{N, I}
                           ) where {T <: AbstractFloat, I <: Unsigned, N}

    @debug "Finding faces corresponding to each segment"
    if !are_materialized_faces(HRPM)
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nγ = length(segment_points)
    bools = fill(false, nγ)
    # Preallocate indices in the most frustrating way
    indices =   [    
                    [ 
                        [ 
                            MVector{N, I}(zeros(I, N)) 
                                for i = 1:length(segment_points[iγ][it])-1 # Segments
                        ] for it = 1:length(segment_points[iγ]) # Tracks
                    ] for iγ = 1:nγ # Angles
                ]
    Threads.@threads for iγ = 1:nγ
        bools[iγ] = find_segment_faces!(segment_points[iγ], indices[iγ], HRPM)
    end
    if !all(bools)
        iγ_bad = findall(x->!x, bools)
        @error "Failed to find indices for some points in segment_points$iγ_bad"
    end
    return indices
end

# Return the face id in which each segment resides
function find_segment_faces(segment_points::Vector{Vector{Vector{Point_2D{T}}}},
                            mesh::UnstructuredMesh_2D{T, I}
                           ) where {T <: AbstractFloat, I <: Unsigned, N}

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
                                  I(0) 
                                      for i = 1:length(segment_points[iγ][it])-1 # Segments
                              ] for it = 1:length(segment_points[iγ]) # Tracks
                          ] for iγ = 1:nγ # Angles
                      ]
    Threads.@threads for iγ = 1:nγ
        bools[iγ] = find_segment_faces_in_angle!(segment_points[iγ], segment_faces[iγ], mesh)
    end
    if !all(bools)
        iγ_bad = findall(x->!x, bools)
        @error "Failed to find segment faces for some points in angles: $iγ_bad"
    end
    return segment_faces
end

# Ray trace an HRPM given the ray spacing and angular quadrature
function ray_trace(tₛ::T,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T}, 
                   HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                   ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    @info "Ray tracing"
    tracks = generate_tracks(tₛ, ang_quad, HRPM)
    segment_points = segmentize(tracks, HRPM)
    nlevels = node_levels(HRPM)
    @info "  - Using the naive segmentize + find face algorithm"
    template_vec = MVector{nlevels, I}(zeros(I, nlevels))
    segment_faces = find_segment_faces(segment_points, HRPM, template_vec)
    return segment_points, segment_faces
end

# Ray trace a mesh given the ray spacing and angular quadrature
function ray_trace(tₛ::T,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T}, 
                   mesh::UnstructuredMesh_2D{T, I}
                   ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    @info "Ray tracing"
    tracks = generate_tracks(tₛ, ang_quad, mesh, boundary_shape = "Rectangle")
    # If the mesh has boundary edges, usue edge-to-edge ray tracing
    if 0 < length(mesh.boundary_edges)
        @info "  - Using the edge-to-edge algorithm"
        return ray_trace_edge_to_edge(tracks, mesh) 
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
function ray_trace_edge_to_edge(tracks::Vector{Vector{LineSegment_2D{T}}},
                                mesh::UnstructuredMesh_2D{T, I}
                                ) where {T <: AbstractFloat, I <: Unsigned}
    if length(mesh.boundary_edges) != 4
        @error "Mesh does not have 4 boundary edges needed for edge-to-edge ray tracing!"
    end
    if length(mesh.materialized_faces) == 0
        @warn "Mesh does not have materialized faces! Ray tracing may be slower."
    end
    if length(mesh.materialized_edges) != 0
        @warn "Mesh has materialized edges! Ray tracing may be slower."
    end
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(tracks)
    segment_points =[
                        [
                            Point_2D{T}[] for it = 1:length(tracks[iγ]) # Tracks 
                        ] for iγ = 1:nγ # Angles
                    ]
    segment_faces = [
                        [
                            I[] for it = 1:length(tracks[iγ]) # Tracks 
                        ] for iγ = 1:nγ # Angles
                    ]
    # For each angle, get the segments and segment faces for each track
    Threads.@threads for iγ = 1:nγ
        ray_trace_angle_edge_to_edge!(tracks[iγ],
                                      segment_points[iγ],
                                      segment_faces[iγ],
                                      mesh)
    end
    return segment_points, segment_faces
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
    nγ = length(tracks)
    segment_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, nγ)
    Threads.@threads for iγ = 1:nγ
        # for each track, intersect the track with the mesh
        segment_points[iγ] = tracks[iγ] .∩ HRPM
    end
    return segment_points
end

function segmentize(tracks::Vector{Vector{LineSegment_2D{T}}},
                    mesh::UnstructuredMesh_2D{T, I}
                    ) where {T <: AbstractFloat, I <: Unsigned}
    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(mesh) 
    @info "    - Segmentizing using the '$int_alg' algorithm"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(tracks)
    segment_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, nγ)
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
