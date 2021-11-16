# Routines for extracting segment/face data for tracks (rays) overlaid on a mesh

# Ray trace an HRPM given the ray spacing and angular quadrature
function ray_trace(tₛ::T,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T}, 
                   HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                   ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    @info "Ray tracing"
    the_tracks = tracks(tₛ, ang_quad, HRPM)
    segment_points = segmentize(the_tracks, HRPM)
    nlevels = levels(HRPM)
    template_vec = MVector{nlevels, I}(zeros(I, nlevels))
    face_indices = find_segment_faces(segment_points, HRPM, template_vec)
    return segment_points, face_indices
end

# Ray trace a mesh given the ray spacing and angular quadrature
function ray_trace(tₛ::T,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T}, 
                   mesh::UnstructuredMesh_2D{T, I}
                   ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    @info "Ray tracing"
    the_tracks = tracks(tₛ, ang_quad, mesh)
    # If the mesh has boundary edges, usue edge-to-edge ray tracing
    if 0 < length(mesh.boundary_edges)
        segment_points = segmentize(the_tracks, mesh)
        face_indices = find_segment_faces(segment_points, mesh)
        return segment_points, face_indices
    else
        segment_points = segmentize(the_tracks, mesh)
        face_indices = find_segment_faces(segment_points, mesh)
        return segment_points, face_indices
    end
end

# Get the segment points and face indices for all tracks in all angles using the
# edge-to-edge ray tracing method. Assumes a rectangular boundary
function ray_trace_edge_to_edge(the_tracks::Vector{Vector{LineSegment_2D{T}}},
                                mesh::UnstructuredMesh_2D{T, I}
                                ) where {T <: AbstractFloat, I <: Unsigned}
    if length(mesh.boundary_edges) != 4
        @error "Mesh does not have 4 boundary edges needed for edge-to-edge ray tracing!"
    end
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(the_tracks)
    seg_points =[
                    [
                        Point_2D{T}[] for it = 1:length(the_tracks[iγ]) # Tracks 
                    ] for iγ = 1:nγ # Angles
                ]
    face_indices =  [
                        [
                            I[] for it = 1:length(the_tracks[iγ]) # Tracks 
                        ] for iγ = 1:nγ # Angles
                    ]
    # For each angle, get the segments and face_indices for each track
    Threads.@threads for iγ = 1:nγ
        ray_trace_edge_to_edge!(the_tracks[iγ],
                                seg_points[iγ],
                                face_indices[iγ],
                                mesh)
    end
    return seg_points, face_indices
end

# Get the segment points and face indices for all tracks in an angle using the
# edge-to-edge ray tracing method. Assumes a rectangular boundary
function ray_trace_edge_to_edge!(the_tracks::Vector{LineSegment_2D{T}},
                                 intersection_points::Vector{Vector{Point_2D{T}}},
                                 face_indices::Vector{Vector{I}},
                                 mesh::UnstructuredMesh_2D{T, I}
                                 ) where {T <: AbstractFloat, I <: Unsigned}
    # For each track, get the intersection points and face_indices
    for it = 1:length(the_tracks)
        (intersection_points[it], face_indices[it]) = ray_trace_edge_to_edge(the_tracks[it], mesh)
    end
end

# Get the segment points and face indices for a single track using the
# edge-to-edge ray tracing method. Assumes a rectangular boundary
function ray_trace_edge_to_edge(l::LineSegment_2D{T},
                                mesh::UnstructuredMesh_2D{T, I}
                                ) where {T <: AbstractFloat, I <: Unsigned}
    # Classify line as intersecting NSEW
    start_point = l.points[1]
    end_point = l.points[2]
    start_nesw = classify_nesw(start_point, mesh)
    end_nesw = classify_nesw(end_point, mesh)
    # Find the starting and ending edges and faces
    start_iedge = get_start_edge_nesw(start_point, mesh.boundary_edges[start_nesw], start_nesw, mesh)
    start_iface = mesh.edge_face_connectivity[start_iedge][2] # 1st entry should be 0
    end_iedge = get_start_edge_nesw(end_point, mesh.boundary_edges[end_nesw], end_nesw, mesh)
    end_iface = mesh.edge_face_connectivity[end_iedge][2] # 1st entry should be 0
    intersection_points = [start_point]
    face_indices = I[]
    if 0 < length(mesh.materialized_edges)
        ray_trace_edge_to_edge_explicit!(l, mesh, intersection_points, face_indices,
                                         start_iedge, start_iface, end_iface)
    else # implicit
        ray_trace_edge_to_edge_implicit!(l, mesh, intersection_points, face_indices,
                                         start_iedge, start_iface, end_iface)
    end
    # The points should already be sorted. We will eliminate any points and face indices 
    # for which the distance between consecutive points is less than the minimum segment length
    if 2 < length(intersection_points)
        # Remove duplicate points
        points_reduced = [intersection_points[1]]
        faces_reduced = I[]
        nipoints = length(intersection_points)
        for i = 2:nipoints
            if minimum_segment_length < distance(last(points_reduced), intersection_points[i])
                push!(points_reduced, intersection_points[i])
                push!(faces_reduced, face_indices[i-1])
            end
        end
        return (points_reduced, faces_reduced) 
    else 
        return (intersection_points, face_indices)
    end  
end

# Linear edges
function ray_trace_edge_to_edge_explicit!(l::LineSegment_2D{T},
                                          mesh::UnstructuredMesh_2D{T, I},
                                          intersection_points::Vector{Point_2D{T}},
                                          face_indices::Vector{I},
                                          start_iedge::I,
                                          start_iface::I,
                                          end_iface::I
                                          ) where {T <: AbstractFloat, I <: Unsigned}
    max_iters = Int64(1E5)
    start_point = l.points[1]
    end_point = l.points[2]
    iedge = start_iedge
    iface = start_iface
    iedge_next = start_iedge
    iface_next = start_iface
    end_reached = false
    iters = 0
#    println("start_iedge: $start_iedge")
#    println("start_iface: $start_iface")
#    f = Figure()
#    display(f)
#    ax = Axis(f[1, 1], aspect = 1)
#    linesegments!(mesh.materialized_edges)
#    linesegments!(l)
    while !end_reached && iters < max_iters
        (iedge_next, iface_next, furthest_point) = next_edge_and_face_explicit(
                                                      iedge, iface,
                                                      l, mesh.materialized_edges,
                                                      mesh.edge_face_connectivity, 
                                                      mesh.face_edge_connectivity)
        # Could not find next face, or jumping back to last face
        if iface_next == iface || (1 < length(face_indices) && iface_next == last(face_indices)) 
            iface_next = next_face_fallback_explicit(iface, last(face_indices), l, mesh)
        else
            push!(intersection_points, furthest_point)
            push!(face_indices, iface)
        end
        iedge = iedge_next 
        iface = iface_next
        # If the furthest intersection is below the minimum segment length to the
        # end point, end here.
        if distance(furthest_point, end_point) < minimum_segment_length
            end_reached = true
            if furthest_point != end_point
                push!(intersection_points, end_point)
                push!(face_indices, end_iface)
            end
        end
#        println("Iteration over")
#        println("")
#        println("")
#        s = readline()
        iters += 1
    end
    if max_iters ≤ iters
        @error "Exceeded max iterations for $l"
    end
end

function next_edge_and_face_explicit(start_iedge::I, start_iface::I, l::LineSegment_2D{T},
                                     materialized_edges::Vector{LineSegment_2D{T}},
                                     edge_face_connectivity::Vector{NTuple{2, I}}, 
                                     face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}}
                                    ) where {T <: AbstractFloat, I <: Unsigned}
    iedge = start_iedge
    iface = start_iface
    iedge_next = start_iedge
    iface_next = start_iface
    start_point = l.points[1]
    # The furthest point along l intersected in this iteration
    furthest_point = start_point
    # For each edge in this face, intersect the track with the edge
    for edge_id in face_edge_connectivity[iface]             
        # If we are testing the edge the ray came in on, skip
#        println("iface: $iface")                          
#        println("edge_id: $edge_id")                      
#        println("furthest_point: ", furthest_point)       
        if edge_id == iedge
#            println("skip")
            continue
        end
        # Edges are linear, so 1 intersection point max
        npoints, point = l ∩ materialized_edges[edge_id]
#        linesegments!(materialized_edges[edge_id])       
#        println("npoints, points: $npoints, ", point)
        # If there's an intersection
        if 0 < npoints 
            # If the intersection point on this edge is further along the ray than the current 
            # furthest point, then we want to leave the face from this edge
            if distance(start_point, furthest_point) ≤ distance(start_point, point)
                furthest_point = point
                # Make sure not to pick the old face id
                if edge_face_connectivity[edge_id][1] == iface
                    iface_next = edge_face_connectivity[edge_id][2]
                else
                    iface_next = edge_face_connectivity[edge_id][1]
                end
                iedge_next = edge_id
#                scatter!(point)
#                println("new furthest point: ", furthest_point)
#                println("Now leaving this face on edge $iedge_next to face $iface_next")
            end
        end
    end
    return iedge_next, iface_next, furthest_point 
end

function next_face_fallback_explicit(current_face::I, last_face::I, 
                                     l::LineSegment_2D{T},
                                     mesh::UnstructuredMesh_2D{T, I},
                                    ) where {T <: AbstractFloat, I <: Unsigned}
    # If the next face could not be determined, or the ray is jumping back to the
    # previous face, this means either:                           
    # (1) The ray is entering or exiting through a vertex, and floating point error 
    #       means the exiting edge did not register an intersection.
    # (2) You're supremely unlucky and a fallback method kicked you to another face
    #       where the next face couldn't be determined
#    @warn "fallback for $l"
    iface_next = current_face
    start_point = l.points[1]
    # The furthest point along l intersected in this iteration
    furthest_point = start_point
    # Check adjacent faces first to see if that is sufficient to solve the problem
    adjacent_faces = get_adjacent_faces(current_face, mesh)
    for iface in adjacent_faces
        npoints, ipoints = l ∩ mesh.materialized_faces[iface]
        if 0 < npoints
            for point in ipoints[1:npoints]
                if distance(start_point, furthest_point) ≤ distance(start_point, point)
                    furthest_point = point
                    iface_next = iface
                end
            end
        end
    end
    if iface_next == current_face || iface_next == last_face
        # If adjacent faces were not sufficient, try all faces sharing the vertices of
        # this face
        # Get the vertex ids for each vertex in the face
        npoints = length(mesh.faces[current_face])
        points = mesh.faces[current_face][2:npoints]
        faces = Set{Int64}()
        for point in points
            union!(faces, faces_sharing_vertex(point, mesh))
        end
        for iface in faces
            npoints, ipoints = l ∩ mesh.materialized_faces[iface]
            if 0 < npoints
                for point in ipoints[1:npoints]
                    if distance(start_point, furthest_point) ≤ distance(start_point, point)
                        furthest_point = point
                        iface_next = iface
                    end
                end
            end
        end
    end
    return I(iface_next)
end

# Linear edges
function ray_trace_edge_to_edge_implicit!(l::LineSegment_2D{T},
                                          mesh::UnstructuredMesh_2D{T, I},
                                          intersection_points::Vector{Point_2D{T}},
                                          face_indices::Vector{I},
                                          start_iedge::I,
                                          start_iface::I,
                                          end_iface::I
                                          ) where {T <: AbstractFloat, I <: Unsigned}
    max_iters = Int64(1E5)
    start_point = l.points[1]
    end_point = l.points[2]
    iedge = start_iedge
    iface = start_iface
    iedge_next = start_iedge
    iface_next = start_iface
    end_reached = false
    iters = 0
    while !end_reached && iters < max_iters
        (iedge_next, iface_next, furthest_point) = next_edge_and_face_implicit(
                                                      iedge, iface,
                                                      l, mesh.points, mesh.edges,
                                                      mesh.edge_face_connectivity, 
                                                      mesh.face_edge_connectivity)
        # Could not find next face, or jumping back to last face
        if iface_next == iface || (1 < length(face_indices) && iface_next == last(face_indices)) 
            iface_next = next_face_fallback_implicit(iface, last(face_indices), l, mesh)
        else
            push!(intersection_points, furthest_point)
            push!(face_indices, iface)
        end
        iedge = iedge_next 
        iface = iface_next
        # If the furthest intersection is below the minimum segment length to the
        # end point, end here.
        if distance(furthest_point, end_point) < minimum_segment_length
            end_reached = true
            if furthest_point != end_point
                push!(intersection_points, end_point)
                push!(face_indices, end_iface)
            end
        end
        iters += 1
    end
    if max_iters ≤ iters
        @error "Exceeded max iterations for $l"
    end
end

# Linear
function next_edge_and_face_implicit(start_iedge::I, start_iface::I, l::LineSegment_2D{T},
                                     points::Vector{Point_2D{T}},
                                     edges::Vector{NTuple{2, I}},
                                     edge_face_connectivity::Vector{NTuple{2, I}}, 
                                     face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}}
                                    ) where {T <: AbstractFloat, I <: Unsigned}
    iedge = start_iedge
    iface = start_iface
    iedge_next = start_iedge
    iface_next = start_iface
    start_point = l.points[1]
    # The furthest point along l intersected in this iteration
    furthest_point = start_point
    # For each edge in this face, intersect the track with the edge
    for edge_id in face_edge_connectivity[iface]             
        # If we are testing the edge the ray came in on, skip
#        println("iface: $iface")                          
#        println("edge_id: $edge_id")                      
#        println("furthest_point: ", furthest_point)       
        if edge_id == iedge
#            println("skip")
            continue
        end
        # Edges are linear, so 1 intersection point max
        npoints, point = l ∩ LineSegment_2D(get_edge_points(points, edge_id))
#        linesegments!(materialized_edges[edge_id])       
#        println("npoints, points: $npoints, ", point)
        # If there's an intersection
        if 0 < npoints 
            # If the intersection point on this edge is further along the ray than the current 
            # furthest point, then we want to leave the face from this edge
            if distance(start_point, furthest_point) ≤ distance(start_point, point)
                furthest_point = point
                # Make sure not to pick the old face id
                if edge_face_connectivity[edge_id][1] == iface
                    iface_next = edge_face_connectivity[edge_id][2]
                else
                    iface_next = edge_face_connectivity[edge_id][1]
                end
                iedge_next = edge_id
#                scatter!(point)
#                println("new furthest point: ", furthest_point)
#                println("Now leaving this face on edge $iedge_next to face $iface_next")
            end
        end
    end
    return iedge_next, iface_next, furthest_point 
end

# linear
function next_face_fallback_implicit(current_face::I, last_face::I, 
                                     l::LineSegment_2D{T},
                                     mesh::UnstructuredMesh_2D{T, I},
                                    ) where {T <: AbstractFloat, I <: Unsigned}
    # If the next face could not be determined, or the ray is jumping back to the
    # previous face, this means either:                           
    # (1) The ray is entering or exiting through a vertex, and floating point error 
    #       means the exiting edge did not register an intersection.
    # (2) You're supremely unlucky and a fallback method kicked you to another face
    #       where the next face couldn't be determined
#    @warn "fallback for $l"
    iface_next = current_face
    start_point = l.points[1]
    # The furthest point along l intersected in this iteration
    furthest_point = start_point
    # Check adjacent faces first to see if that is sufficient to solve the problem
    adjacent_faces = get_adjacent_faces(current_face, mesh)
    for iface in adjacent_faces
        type_id = mesh.faces[iface][1]
        if type_id == 5 # Triangle
            npoints, ipoints = l ∩ Triangle_2D(
                                    get_face_points(mesh, 
                                                    face::NTuple{4, I})::NTuple{3, Point_2D{T}}
                                  )
            if 0 < npoints
                for point in ipoints[1:npoints]
                    if distance(start_point, furthest_point) ≤ distance(start_point, point)
                        furthest_point = point
                        iface_next = iface
                    end
                end
            end
        elseif type_id == 9 # Quadrilateral
            npoints, ipoints = l ∩ Quadrilateral_2D(
                                    get_face_points(mesh, 
                                                    face::NTuple{5, I})::NTuple{4, Point_2D{T}}
                                   )
            if 0 < npoints
                for point in ipoints[1:npoints]
                    if distance(start_point, furthest_point) ≤ distance(start_point, point)
                        furthest_point = point
                        iface_next = iface
                    end
                end
            end
        end
    end
    if iface_next == current_face || iface_next == last_face
        # If adjacent faces were not sufficient, try all faces sharing the vertices of
        # this face
        # Get the vertex ids for each vertex in the face
        npoints = length(mesh.faces[current_face])
        points = mesh.faces[current_face][2:npoints]
        faces = Set{Int64}()
        for point in points
            union!(faces, faces_sharing_vertex(point, mesh))
        end
        for iface in faces
            npoints, ipoints = l ∩ mesh.materialized_faces[iface]
            if 0 < npoints
                for point in ipoints[1:npoints]
                    if distance(start_point, furthest_point) ≤ distance(start_point, point)
                        furthest_point = point
                        iface_next = iface
                    end
                end
            end
        end
    end
    return I(iface_next)
end

function segmentize(the_tracks::Vector{Vector{LineSegment_2D{T}}},
                    HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                    ) where {T <: AbstractFloat, I <: Unsigned}

    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(HRPM) 
    @info "Segmentizing using the '$int_alg' algorithm"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(the_tracks)
    seg_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, nγ)
    Threads.@threads for iγ = 1:nγ
        # for each track, intersect the track with the mesh
        seg_points[iγ] = the_tracks[iγ] .∩ HRPM
    end
    return seg_points
end

function segmentize(the_tracks::Vector{Vector{LineSegment_2D{T}}},
                    mesh::UnstructuredMesh_2D{T, I}
                    ) where {T <: AbstractFloat, I <: Unsigned}
    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(mesh) 
    @info "Segmentizing using the '$int_alg' algorithm"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    nγ = length(the_tracks)
    seg_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, nγ)
    Threads.@threads for iγ = 1:nγ
        # for each track, intersect the track with the mesh
        seg_points[iγ] = the_tracks[iγ] .∩ mesh
    end
    return seg_points
end

function find_segment_faces(seg_points::Vector{Vector{Vector{Point_2D{T}}}},
                            HRPM::HierarchicalRectangularlyPartitionedMesh{T, I},
                            template_vec::MVector{N, I}
                           ) where {T <: AbstractFloat, I <: Unsigned, N}

    @debug "Finding faces corresponding to each segment"
    if !are_materialized_faces(HRPM)
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nγ = length(seg_points)
    bools = fill(false, nγ)
    # Preallocate indices in the most frustrating way
    indices =   [    
                    [ 
                        [ 
                            MVector{N, I}(zeros(I, N)) 
                                for i = 1:length(seg_points[iγ][it])-1 # Segments
                        ] for it = 1:length(seg_points[iγ]) # Tracks
                    ] for iγ = 1:nγ # Angles
                ]
    Threads.@threads for iγ = 1:nγ
        bools[iγ] = find_segment_faces!(seg_points[iγ], indices[iγ], HRPM)
    end
    if !all(bools)
        iγ_bad = findall(x->!x, bools)
        @error "Failed to find indices for some points in seg_points$iγ_bad"
    end
    return indices
end

# Get the face indices for all tracks in a single angle
function find_segment_faces!(points::Vector{Vector{Point_2D{T}}},
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
        bools[it] = find_segment_faces!(points[it], indices[it], HRPM)
    end
    return all(bools)
end

# Get the face indices for all segments in a single track
function find_segment_faces!(points::Vector{Point_2D{T}},
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

function find_segment_faces(seg_points::Vector{Vector{Vector{Point_2D{T}}}},
                            mesh::UnstructuredMesh_2D{T, I}
                           ) where {T <: AbstractFloat, I <: Unsigned, N}

    @debug "Finding faces corresponding to each segment"
    if !(0 < length(mesh.materialized_faces))
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nγ = length(seg_points)
    bools = fill(false, nγ)
    # Preallocate indices in the most frustrating way
    indices =   [    
                    [ 
                        [ 
                            I(0) 
                                for i = 1:length(seg_points[iγ][it])-1 # Segments
                        ] for it = 1:length(seg_points[iγ]) # Tracks
                    ] for iγ = 1:nγ # Angles
                ]
    Threads.@threads for iγ = 1:nγ
        bools[iγ] = find_segment_faces!(seg_points[iγ], indices[iγ], mesh)
    end
    if !all(bools)
        iγ_bad = findall(x->!x, bools)
        @error "Failed to find indices for some points in seg_points$iγ_bad"
    end
    return indices
end

# Get the face indices for all tracks in a single angle
function find_segment_faces!(points::Vector{Vector{Point_2D{T}}},
                             indices::Vector{Vector{I}},
                             mesh::UnstructuredMesh_2D{T, I}
                            ) where {T <: AbstractFloat, I <: Unsigned, N}
    nt = length(points)
    bools = fill(false, nt)
    # for each track, find the segment indices
    for it = 1:nt
        # Points in the track
        npoints = length(points[it])
        # Returns true if indices were found for all segments in the track
        bools[it] = find_segment_faces!(points[it], indices[it], mesh)
    end
    return all(bools)
end

# Get the face indices for all segments in a single track
function find_segment_faces!(points::Vector{Point_2D{T}},
                             indices::Vector{I},
                             mesh::UnstructuredMesh_2D{T, I}
                            ) where {T <: AbstractFloat, I <: Unsigned, N}
    # Points in the track
    npoints = length(points)
    bools = fill(false, npoints-1)
    # Test the midpoint of each segment to find the face
    for iseg = 1:npoints-1
        p_midpoint = midpoint(points[iseg], points[iseg+1])
        indices[iseg] = find_face(p_midpoint, mesh)
        bools[iseg] = 0 < indices[iseg] 
    end
    return all(bools)
end

# Plot
# -------------------------------------------------------------------------------------------------
# Plot ray tracing data one angle at a time.
function linesegments!(seg_points::Vector{Vector{Vector{Point_2D{T}}}},
                       seg_faces::Vector{Vector{Vector{I}}}) where {T <: AbstractFloat, I <: Unsigned} 
    println("Press enter to plot the segments in the next angle")
    colormap = ColorSchemes.tab20.colors
    lines_by_color = Vector{Vector{LineSegment_2D{T}}}(undef, 20)
    nγ = length(seg_points)
    for iγ = 1:nγ
        for icolor = 1:20
            lines_by_color[icolor] = LineSegment_2D{T}[]
        end
        for it = 1:length(seg_points[iγ])
            for iseg = 1:length(seg_points[iγ][it])-1
                l = LineSegment_2D(seg_points[iγ][it][iseg], seg_points[iγ][it][iseg+1]) 
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
