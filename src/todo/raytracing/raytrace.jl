# # Routines for extracting segment/face data for tracks (rays) overlaid on a mesh
# 
# # Return the HRPM/face indices in which each segment resides
# # Type-stable, other than warning
# function find_segment_faces(segment_points::Vector{Vector{Vector{Point_2D{F}}}},
#                             HRPM::HierarchicalRectangularlyPartitionedMesh{F, U},
#                             template_vec::MVector{N, U}
#                            ) where {F <: AbstractFloat, U <: Unsigned, N}
# 
#     @info "    - Finding faces for each segment"
#     if !has_materialized_faces(HRPM)
#         @warn "Faces are not materialized for this mesh. This will be VERY slow"
#     end
#     nγ = length(segment_points)
#     bools = fill(false, nγ)
#     # Preallocate indices in the most frustrating way
#     indices =   [    
#                     [ 
#                         [ 
#                             MVector{N, U}(zeros(U, N)) 
#                                 for i = 1:length(segment_points[iγ][it])-1 # Segments
#                         ] for it = 1:length(segment_points[iγ]) # Tracks
#                     ] for iγ = 1:nγ # Angles
#                 ]
#     Threads.@threads for iγ = 1:nγ
#         bools[iγ] = find_segment_faces_in_angle!(segment_points[iγ], indices[iγ], HRPM)
#     end
#     if !all(bools)
#         iγ_bad = findall(x->!x, bools)
#         @error "Failed to find indices for some points in segment_points$iγ_bad"
#     end
#     return indices
# end
# 
# Return the face id in which each segment resides
function find_segment_faces(segment_points::Vector{Vector{Vector{Point_2D}}},
                            mesh::UnstructuredMesh_2D)
    @info "    - Finding faces for each segment"
    if 0 == length(mesh.materialized_faces)
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nγ = length(segment_points)
    # Preallocate indices in the most frustrating way
    segment_faces = [[fill(UInt32(0), length(segment_points[iγ][it]) - 1) # Segments
                      for it in 1:length(segment_points[iγ])] for iγ in 1:nγ]
    Threads.@threads for iγ in 1:nγ
        find_segment_faces_in_angle!(segment_points[iγ], segment_faces[iγ], mesh)
        for it in 1:length(segment_points[iγ])
            if any(x -> x == 0x00000000, segment_faces[iγ][it])
                @warn "Segment face not found for face in [$iγ][$it]"
            end
        end
    end
    return segment_faces
end

# # Ray trace an HRPM given the ray spacing and angular quadrature
# # Not type-stable
# function ray_trace(tₛ::F,
#                    ang_quad::ProductAngularQuadrature{nᵧ, nₚ, F}, 
#                    HRPM::HierarchicalRectangularlyPartitionedMesh{F, U}
#                    ) where {nᵧ, nₚ, F <: AbstractFloat, U <: Unsigned}
#     @info "Ray tracing"
#     tracks = generate_tracks(tₛ, ang_quad, HRPM)
#     segment_points = segmentize(tracks, HRPM)
#     ind_size = node_height(HRPM) + 1
#     @info "  - Using the naive segmentize + find face algorithm"
#     template_vec = MVector{ind_size, U}(zeros(U, ind_size))
#     segment_faces = find_segment_faces(segment_points, HRPM, template_vec)
#     return segment_points, segment_faces
# end
# 
# Ray trace a mesh given the ray spacing and angular quadrature
function ray_trace(tₛ::Float64,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ},
                   mesh::UnstructuredMesh_2D) where {nᵧ, nₚ}
    @info "Ray tracing"
    tracks = generate_tracks(tₛ, ang_quad, mesh, boundary_shape = "Rectangle")
    if use_E2E_raytracing(mesh)
        @info "  - Using the edge-to-edge algorithm"
        segment_points, segment_faces = ray_trace_edge_to_edge(tracks, mesh)
        validate_ray_tracing_data(segment_points, segment_faces, mesh,
                                  plot = enable_visualization)
        return segment_points, segment_faces
    else
        @info "  - Using the naive segmentize + find face algorithm"
        segment_points = segmentize(tracks, mesh)
        segment_faces = find_segment_faces(segment_points, mesh)
        validate_ray_tracing_data(segment_points, segment_faces, mesh,
                                  plot = enable_visualization)
        return segment_points, segment_faces
    end
end

# Get the segment points and the face which the segment lies in for all segments, 
# in all tracks, in all angles, using the edge-to-edge ray tracing method. 
# Assumes a rectangular boundary
function ray_trace_edge_to_edge(tracks::Vector{Vector{LineSegment_2D}},
                                mesh::UnstructuredMesh_2D)
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
    segment_points = [[Point_2D[] for it in 1:length(tracks[iγ])] for iγ in 1:nγ]
    segment_faces = [[UInt32[] for it in 1:length(tracks[iγ])] for iγ in 1:nγ]
    # For each angle, get the segments and segment faces for each track
    Threads.@threads for iγ in 1:nγ
        ray_trace_angle_edge_to_edge!(tracks[iγ],
                                      segment_points[iγ],
                                      segment_faces[iγ],
                                      mesh)
    end
    return segment_points, segment_faces
end

# # Return the points of intersection between the tracks and the mesh edges.
# # Returns a Vector{Vector{Vector{Point_2D{F}}}}.
# #   index 1 = γ
# #   index 2 = track
# #   index 3 = point/segment
# # Type-stable other than the error message
# function segmentize(tracks::Vector{Vector{LineSegment_2D{F}}},
#                     HRPM::HierarchicalRectangularlyPartitionedMesh{F, U}
#                     ) where {F <: AbstractFloat, U <: Unsigned}
# 
#     # Give info about intersection algorithm being used
#     int_alg = get_intersection_algorithm(HRPM) 
#     @info "    - Segmentizing using the '$int_alg' algorithm"
#     # index 1 = γ
#     # index 2 = track
#     # index 3 = point/segment
#     nγ = length(tracks)
#     segment_points = Vector{Vector{Vector{Point_2D{F}}}}(undef, nγ)
#     Threads.@threads for iγ = 1:nγ
#         # for each track, intersect the track with the mesh
#         segment_points[iγ] = tracks[iγ] .∩ HRPM
#     end
#     return segment_points
# end
# 
# Return the points of intersection between the tracks and the mesh edges.
# Returns a Vector{Vector{Vector{Point_2D{F}}}}.
#   index 1 = γ
#   index 2 = track
#   index 3 = point/segment
function segmentize(tracks::Vector{Vector{LineSegment_2D}}, mesh::UnstructuredMesh_2D)
    # Give info about intersection algorithm being used
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(tracks)
    segment_points = Vector{Vector{Vector{Point_2D}}}(undef, nγ)
    if length(mesh.materialized_edges) !== 0
        @info "    - Segmentizing using the 'Edges - Explicit' algorithm"
        Threads.@threads for iγ in 1:nγ
            segment_points[iγ] = intersect_edges_explicit.(tracks[iγ],
                                                           Ref(mesh.materialized_edges))
        end
    elseif length(mesh.edges) !== 0
        @info "    - Segmentizing using the 'Edges - Implicit' algorithm"
        Threads.@threads for iγ in 1:nγ
            segment_points[iγ] = intersect_edges_implicit.(tracks[iγ], Ref(mesh.edges),
                                                           Ref(mesh.points))
        end
    elseif length(mesh.materialized_faces) !== 0
        @info "    - Segmentizing using the 'Faces - Explicit' algorithm"
        Threads.@threads for iγ in 1:nγ
            segment_points[iγ] = intersect_faces_explicit.(tracks[iγ],
                                                           Ref(mesh.materialized_faces))
        end
    else
        @info "    - Segmentizing using the 'Faces - Implicit' algorithm"
        Threads.@threads for iγ in 1:nγ
            segment_points[iγ] = intersect_faces_implicit.(tracks[iγ], Ref(mesh.faces),
                                                           Ref(mesh.points))
        end
    end
    return segment_points
end

# Assumes materialized faces
function validate_ray_tracing_data(segment_points::Vector{Vector{Vector{Point_2D}}},
                                   segment_faces::Vector{Vector{Vector{UInt32}}},
                                   mesh::UnstructuredMesh_2D;
                                   plot::Bool = false)
    @info "  - Validating ray tracing data"
    nthreads = Threads.nthreads()
    nsegs = zeros(Int64, nthreads)
    nsegs_problem = zeros(Int64, nthreads) # Problem segment if the face doesn't match, and it's over 10 μm
    problem_indices = [SVector{3, Int64}[] for i in 1:nthreads]
    plot_segs_face = [LineSegment_2D[] for i in 1:nthreads]
    plot_points_face = [Point_2D[] for i in 1:nthreads]

    # Validate faces
    Threads.@threads for iγ in 1:length(segment_faces)
        for it in 1:length(segment_faces[iγ])
            nsegs[Threads.threadid()] += length(segment_faces[iγ][it])
            for iseg in 1:length(segment_faces[iγ][it])
                p_midpoint = midpoint(segment_points[iγ][it][iseg],
                                      segment_points[iγ][it][iseg + 1])
                face = segment_faces[iγ][it][iseg]
                if p_midpoint ∉ mesh.materialized_faces[face]
                    p1 = segment_points[iγ][it][iseg]
                    p2 = segment_points[iγ][it][iseg + 1]
                    l = LineSegment_2D(p1, p2)
                    problem_length = 1e-3 < arclength(l)
                    if problem_length
                        nsegs_problem[Threads.threadid()] += 1
                        push!(problem_indices[Threads.threadid()], SVector(iγ, it, iseg))
                    end
                    # Append the points and line if we want to plot them.
                    # We only want to plot if actually a problem.
                    if enable_visualization && plot && problem_length
                        append!(plot_points_face[Threads.threadid()], [p1, p2])
                        push!(plot_segs_face[Threads.threadid()], l)
                    end
                end
            end
        end
    end

    # Attempt to fix problem segments if used E2E ray tracing
    if use_E2E_raytracing(mesh)
        @info "    - Attempting to fix $(sum(nsegs_problem)) segments with incorrect faces"
        fixed = [Int64[] for i in 1:nthreads]
        Threads.@threads for i in 1:nthreads
            for (iprob, problem_index) in enumerate(problem_indices[i])
                # Ray trace the track in reverse, then test to see if that fixed things
                new_info_ok = true
                iγ, it, iseg = problem_index
                p_midpoint = midpoint(segment_points[iγ][it][iseg],
                                      segment_points[iγ][it][iseg + 1])
                face = segment_faces[iγ][it][iseg]
                # Check to see if still a problem. May have been fixed by a previous segment updating the 
                # points/faces
                if p_midpoint ∈ mesh.materialized_faces[face]
                    nsegs_problem[i] -= 1
                    continue
                end
                npoints = length(segment_points[iγ][it])
                problem_line = LineSegment_2D(segment_points[iγ][it][1],
                                              segment_points[iγ][it][npoints])
                line_reversed = LineSegment_2D(problem_line.points[2],
                                               problem_line.points[1])
                reversed_points, reversed_faces = ray_trace_track_edge_to_edge(line_reversed,
                                                                               mesh)
                # Check
                for iseg in 1:length(reversed_faces)
                    p_midpoint = midpoint(reversed_points[iseg], reversed_points[iseg + 1])
                    face = reversed_faces[iseg]
                    if p_midpoint ∉ mesh.materialized_faces[face]
                        l = LineSegment_2D(reversed_points[iseg], reversed_points[iseg + 1])
                        problem_length = 1e-3 < arclength(l)
                        if problem_length
                            new_info_ok = false
                        end
                    end
                end
                if new_info_ok
                    push!(fixed[i], iprob)
                    nsegs_problem[i] -= 1
                    segment_points[iγ][it] = reverse(reversed_points)
                    segment_faces[iγ][it] = reverse(reversed_faces)
                else
                    new_face = findface(p_midpoint, mesh)
                    if new_face == 0
                        @warn "Could not find new segment face for segment $(problem_index[i])"
                    else
                        segment_faces[iγ][it][iseg] = new_face
                    end
                end
            end
        end
    end

    # Visualize
    problem_segs = sum(nsegs_problem)
    if enable_visualization && plot && 0 < problem_segs
        f = Figure()
        ax = Axis(f[1, 1], aspect = 1)
        display(f)
        mesh!(mesh.materialized_faces, color = (:black, 0.15))
        if 0 < length(mesh.materialized_edges)
            linesegments!(mesh.materialized_edges, color = :blue)
        else
            linesegments!(materialize_edges(mesh), color = :blue)
        end
        for i in 1:nthreads
            if 0 < length(fixed[i])
                deleteat!(plot_segs_face[i], fixed[i])
            end
            if 0 < length(plot_segs_face[i])
                linesegments!(plot_segs_face[i], color = :red)
            end
            if 0 < length(fixed[i])
                deleteat!(plot_points_face[i],
                          sort!(vcat(2 .* fixed[i], 2 .* fixed[i] .- 1)))
            end
            if 0 < length(plot_points_face[i])
                scatter!(plot_points_face[i], color = :red)
            end
        end
    end
    nsegs_total = sum(nsegs)
    prob_percent = 100 * problem_segs / nsegs_total
    @info "    - Problem segments: $problem_segs, Total segments: $nsegs_total"
    if 0 < problem_segs
        @info "      - Check for mesh overlap (darker gray) around areas of many problem segments."
    end
    return problem_segs == 0
end

# Plot
# -------------------------------------------------------------------------------------------------
# Plot ray tracing data one angle at a time.
if enable_visualization
    #    function linesegments!(segment_points::Vector{Vector{Vector{Point_2D}}},
    #                           segment_faces::Vector{Vector{Vector{UInt32}}})
    #        println("Press enter to plot the segments in the next angle")
    #        colormap = ColorSchemes.tab20.colors
    #        lines_by_color = Vector{Vector{LineSegment_2D{T}}}(undef, 20)
    #        nγ = length(segment_points)
    #        for iγ = 1:nγ
    #            f = Figure()
    #            ax = Axis(f[1, 1], aspect = 1)
    #            display(f)
    #            for icolor = 1:20
    #                lines_by_color[icolor] = LineSegment_2D{T}[]
    #            end
    #            for it = 1:length(segment_points[iγ])
    #                for iseg = 1:length(segment_points[iγ][it])-1
    #                    l = LineSegment_2D(segment_points[iγ][it][iseg], segment_points[iγ][it][iseg+1]) 
    #                    face = segment_faces[iγ][it][iseg]
    #                    if face == 0
    #                        @error "Segment [$iγ][$it][$iseg] has a face id of 0"
    #                    end
    #                    push!(lines_by_color[face % 20 + 1], l)
    #                end
    #            end
    #            for icolor = 1:20
    #                linesegments!(lines_by_color[icolor], color = colormap[icolor])
    #            end
    #            s = readline()
    #            println(iγ)
    #        end
    #    end

    # Set visualize_ray_tracing = true in constants.jl to get this to work.
    function plot_track_edge_to_edge(track::LineSegment_2D, mesh::UnstructuredMesh_2D)
        @info "Plotting ray tracing of track. Press enter to advance the ray"
        f = Figure()
        ax = Axis(f[1, 1], aspect = 1)
        display(f)
        linesegments!(mesh.materialized_edges, color = :blue)
        linesegments!(track, color = :orange)
        segment_points, segment_faces = ray_trace_track_edge_to_edge(track, mesh)
        return segment_points, segment_faces
    end
end
