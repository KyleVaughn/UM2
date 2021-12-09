# Routines for extracting segment/face data for tracks (rays) overlaid on a mesh
num_fallback = 0
num_fallback_adjacent = 0
num_fallback_vertex = 0
num_fallback_vertex2 = 0

# Return the HRPM/face indices in which each segment resides
# Type-stable, other than warning
function find_segment_faces(segment_points::Vector{Vector{Vector{Point_2D{F}}}},
                            HRPM::HierarchicalRectangularlyPartitionedMesh{F, U},
                            template_vec::MVector{N, U}
                           ) where {F <: AbstractFloat, U <: Unsigned, N}

    @info "    - Finding faces for each segment"
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
    if !all(bools)
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
    if !all(bools)
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
    global num_fallback = 0
    global num_fallback_adjacent = 0
    global num_fallback_vertex = 0
    global num_fallback_vertex2 = 0
    tracks = generate_tracks(tₛ, ang_quad, mesh, boundary_shape = "Rectangle")
    has_mat_edges = 0 < length(mesh.materialized_edges)
    has_mat_faces = 0 < length(mesh.materialized_faces)
    # If the mesh has boundary edges and materialized edges and faces, 
    # use edge-to-edge ray tracing
    if 0 < length(mesh.boundary_edges) && has_mat_edges && has_mat_faces 
        @info "  - Using the edge-to-edge algorithm"
        segment_points, segment_faces = ray_trace_edge_to_edge(tracks, mesh) 
        @info "    - Segments needing fallback     : $num_fallback"
        @info "      - Adjacent faces fallback     : $num_fallback_adjacent"
        @info "      - Shared vertices fallback    : $num_fallback_vertex"
        @info "      - Shared vertices 2 fallback  : $num_fallback_vertex2"
        validate_ray_tracing_data(segment_points, segment_faces, mesh)
        return segment_points, segment_faces
    else
        @info "  - Using the naive segmentize + find face algorithm"
        segment_points = segmentize(tracks, mesh)
        segment_faces = find_segment_faces(segment_points, mesh)
        validate_ray_tracing_data(segment_points, segment_faces, mesh)
        return segment_points, segment_faces
    end
end

# Get the segment points and the face which the segment lies in for all segments, 
# in all tracks, in all angles, using the edge-to-edge ray tracing method. 
# Assumes a rectangular boundary
function ray_trace_edge_to_edge(tracks::Vector{Vector{LineSegment_2D{F}}},
                                mesh::UnstructuredMesh_2D{F, U}
                                ) where {F <: AbstractFloat, U <: Unsigned}
    if visualize_ray_tracing
        @error "visualize_ray_tracing = true. Please reset to false in constants.jl"
    end
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
                                      mesh.points,
                                      mesh.edges,
                                      mesh.materialized_edges,
                                      mesh.faces,
                                      mesh.materialized_faces,
                                      mesh.edge_face_connectivity,
                                      mesh.face_edge_connectivity,
                                      mesh.boundary_edges
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
    @info "    - Segmentizing using the '$int_alg' algorithm"
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

function validate_ray_tracing_data(segment_points::Vector{Point_2D{F}},
                                   segment_faces::Vector{U},
                                   mesh::UnstructuredMesh_2D{F, U};
                                   plot::Bool = false,
                                   debug::Bool = false
                                  ) where {F <: AbstractFloat, U <: Unsigned} 
    @info "  - Validating ray tracing data"
    # Check that all segment faces are correct
    nsegs_problem = 0 # Problem segment if the face doesn't match, and it's over 10 μm
    plot_segs = LineSegment_2D{F}[]
    plot_points = Point_2D{F}[]
    if enable_visualization && plot
        f = Figure()
        ax = Axis(f[1, 1], aspect = 1)
        display(f)
        linesegments!(mesh.materialized_edges, color = :blue)
    end
    nsegs = length(segment_faces) - 1
    for iseg = 1:nsegs
        p_midpoint = midpoint(segment_points[iseg], segment_points[iseg+1])
        face = segment_faces[iseg]
        if p_midpoint ∉  mesh.materialized_faces[face]
            p1 = segment_points[iseg]
            p2 = segment_points[iseg + 1]
            l = LineSegment_2D(p1, p2)
            problem_length = 1e-3 < arc_length(l)
            if debug || problem_length 
                @warn "Face mismatch for segment [$iseg]\n" * 
                      "   Reference face: $(find_face(p_midpoint, mesh)), Face: $face"
                nsegs_problem += 1
            end
            # append the points, line if we want to plot them
            # we only want to plot if actually a problem, or if debug is on.
            if enable_visualization && plot && (debug || problem_length)
                append!(plot_points, [p1, p2])
                push!(plot_segs, l)
            end
        end
    end
    if enable_visualization && plot
        if 0 < length(plot_segs)
            linesegments!(plot_segs, color = :red)
        end
        if 0 < length(plot_points)
            scatter!(plot_points, color = :red)
        end
    end
    prob_percent = 100*nsegs_problem/nsegs
    @info "    - Segments: $nsegs, Problem segments: $nsegs_problem"
    if nsegs_problem == 0
        @info "    - Problem %: $prob_percent, or approx 1 in ∞"
    else
        @info "    - Problem %: $prob_percent, or approx 1 in $(Int64(ceil(100/prob_percent)))"
    end

    return nsegs_problem == 0
end

function validate_ray_tracing_data(segment_points::Vector{Vector{Vector{Point_2D{F}}}},
                                   segment_faces::Vector{Vector{Vector{U}}},
                                   mesh::UnstructuredMesh_2D{F, U};
                                   plot::Bool = false,
                                   debug::Bool = false
                                  ) where {F <: AbstractFloat, U <: Unsigned} 
    @info "  - Validating ray tracing data"
    # Check that all segment faces are correct
    nthreads = Threads.nthreads()
    nsegs = zeros(Int64, nthreads)
    nsegs_problem = zeros(Int64, nthreads) # Problem segment if the face doesn't match, and it's over 10 μm
    nsegs_potential_problem = zeros(Int64, nthreads) # Might be a problem segment
    problem_indices = [ SVector{3, Int64}[] for i = 1:nthreads ]
    plot_segs_face = [ LineSegment_2D{F}[] for i = 1:nthreads ] 
    plot_points_face = [ Point_2D{F}[] for i = 1:nthreads ]
    plot_segs_conn = [ LineSegment_2D{F}[] for i = 1:nthreads ] 
    plot_points_conn = [ Point_2D{F}[] for i = 1:nthreads ]
    if enable_visualization && plot
        f = Figure()
        ax = Axis(f[1, 1], aspect = 1)
        display(f)
        linesegments!(mesh.materialized_edges, color = :blue)
    end

    # Validate faces
    @info "    - Validating segment faces"
    nγ = length(segment_faces)
    Threads.@threads for iγ = 1:nγ
        for it = 1:length(segment_faces[iγ])
            nsegs[Threads.threadid()] += length(segment_faces[iγ][it])
            for iseg = 1:length(segment_faces[iγ][it])
                p_midpoint = midpoint(segment_points[iγ][it][iseg], segment_points[iγ][it][iseg+1])
                face = segment_faces[iγ][it][iseg]
                if p_midpoint ∉  mesh.materialized_faces[face]
                    p1 = segment_points[iγ][it][iseg]
                    p2 = segment_points[iγ][it][iseg + 1]
                    l = LineSegment_2D(p1, p2)
                    problem_length = 1e-3 < arc_length(l)
                    if debug || problem_length 
                        @warn "Face mismatch for segment [$iγ][$it][$iseg]\n" * 
                              "   Reference face: $(find_face(p_midpoint, mesh)), Face: $face"
                        nsegs_problem[Threads.threadid()] += 1
                    end
                    # append the points, line if we want to plot them
                    # we only want to plot if actually a problem, or if debug is on.
                    if enable_visualization && plot && (debug || problem_length)
                        append!(plot_points_face[Threads.threadid()], [p1, p2])
                        push!(plot_segs_face[Threads.threadid()], l)
                    end
                end
            end
        end
    end

    # Validate connectivity
    @info "    - Validating segment faces"
    Threads.@threads for iγ = 1:nγ
        for it = 1:length(segment_faces[iγ])
            for iseg = 1:length(segment_faces[iγ][it])-1
                f1 = mesh.faces[segment_faces[iγ][it][iseg]]
                f2 = mesh.faces[segment_faces[iγ][it][iseg + 1]]
                has_shared_vertex = false
                for i = 2:length(f1), j = 2:length(f2)
                    if f1[i] === f2[j]
                        has_shared_vertex = true
                        break
                    end
                end
                if !(has_shared_vertex)
                    p1 = segment_points[iγ][it][iseg]
                    p2 = segment_points[iγ][it][iseg + 1]
                    p3 = segment_points[iγ][it][iseg + 2]
                    l1 = LineSegment_2D(p1, p2)
                    l2 = LineSegment_2D(p2, p3)
                    if debug
                        @warn "Potential connectivity problem for segments [$iγ][$it]([$iseg], [$(iseg + 1)])"
                    end
                    nsegs_potential_problem[Threads.threadid()] += 1
                    # append the points, lines if we want to plot them
                    if enable_visualization && plot
                        append!(plot_points_conn[Threads.threadid()], [p1, p2, p3])
                        append!(plot_segs_conn[Threads.threadid()], [l1, l2])
                    end
                end
            end
        end
    end

    # Visualize
    if enable_visualization && plot
        for i = 1:nthreads
            if 0 < length(plot_segs_conn[i])
                linesegments!(plot_segs_conn[i], color = :yellow)
            end
            if 0 < length(plot_points_conn[i])
                scatter!(plot_points_conn[i], color = :yellow)
            end
            if 0 < length(plot_segs_face[i])
                linesegments!(plot_segs_face[i], color = :red)
            end
            if 0 < length(plot_points_face[i])
                scatter!(plot_points_face[i], color = :red)
            end
        end
    end
    problem_segs = sum(nsegs_problem)
    potential_problem_segs = sum(nsegs_potential_problem)
    nsegs_total = sum(nsegs)
    prob_percent = 100*problem_segs/nsegs_total
    potential_prob_percent = 100*potential_problem_segs/nsegs_total
    @info "    - Total segments: $nsegs_total"
    @info "    - Problem segments: $problem_segs"
    if problem_segs == 0
        @info "    - Problem %: $prob_percent, or approx 1 in ∞"
    else
        @info "    - Problem %: $prob_percent, or approx 1 in $(Int64(ceil(100/prob_percent)))"
    end
    @info "    - Potential problem segments: $potential_problem_segs"
    if potential_problem_segs == 0
        @info "    - Potential problem %: $potential_prob_percent, or approx 1 in ∞"
    else
        @info "    - Potential problem %: $potential_prob_percent, or approx 1 in $(Int64(ceil(100/prob_percent)))"
    end


    return problem_segs == 0
end

# Plot
# -------------------------------------------------------------------------------------------------
# Plot ray tracing data one angle at a time.
if enable_visualization
    function linesegments!(segment_points::Vector{Vector{Vector{Point_2D{T}}}},
                           seg_faces::Vector{Vector{Vector{I}}}) where {T <: AbstractFloat, I <: Unsigned} 
        println("Press enter to plot the segments in the next angle")
        colormap = ColorSchemes.tab20.colors
        lines_by_color = Vector{Vector{LineSegment_2D{T}}}(undef, 20)
        nγ = length(segment_points)
        for iγ = 1:nγ
            f = Figure()
            ax = Axis(f[1, 1], aspect = 1)
            display(f)
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
    
    # Set visualize_ray_tracing = true in constants.jl to get this to work.
    function plot_track_edge_to_edge(track::LineSegment_2D{F},
                                      mesh::UnstructuredMesh_2D{F, U} 
                                    ) where {F <: AbstractFloat, U <: Unsigned}
        @info "Plotting ray tracing of track. Press enter to advance the ray"
        f = Figure()
        ax = Axis(f[1, 1], aspect = 1)
        display(f)
        linesegments!(mesh.materialized_edges, color = :blue)
        linesegments!(track, color = :orange)
        segment_points, segment_faces = ray_trace_track_edge_to_edge(track,
                                                                mesh.points,
                                                                mesh.edges,
                                                                mesh.materialized_edges,
                                                                mesh.faces,
                                                                mesh.materialized_faces,
                                                                mesh.edge_face_connectivity,
                                                                mesh.face_edge_connectivity,
                                                                mesh.boundary_edges)
        return segment_points, segment_faces
    end
end
