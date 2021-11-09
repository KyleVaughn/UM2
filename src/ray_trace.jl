# @code_warntype checked 2021/11/09

# Classify a point as on the North, East, South, or West edge of a rectangular mesh
function classify_NESW(p::Point_2D{T}, 
                       mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    y_N = mesh.points[mesh.edges[mesh.boundary_edges[1][1]][1]].x[2] 
    x_E = mesh.points[mesh.edges[mesh.boundary_edges[2][1]][1]].x[1]
    y_S = mesh.points[mesh.edges[mesh.boundary_edges[3][1]][1]].x[2]
    x_W = mesh.points[mesh.edges[mesh.boundary_edges[4][1]][1]].x[1]
    if abs(p.x[2] - y_N) < 1e-4
        return 1 # North
    elseif abs(p.x[1] - x_E) < 1e-4
        return 2 # East
    elseif abs(p.x[2] - y_S) < 1e-4
        return 3 # South
    elseif abs(p.x[1] - x_W) < 1e-4
        return 4 # West
    else
        @error "Could not classify point"
        return 0 # Error
    end
end

# Get the boundary edge that a point lies on for a rectangular mesh
function get_start_edge_NESW(p::Point_2D{T}, 
                             boundary_edge_indices::Vector{I},
                             NESW::Int64,
                             mesh::UnstructuredMesh_2D{T, I}
                             ) where {T <: AbstractFloat, I <: Unsigned}
    if NESW == 1 || NESW == 3
        # On North or South edge. Just check x coordinates
        xₚ = p.x[1]
        for iedge in boundary_edge_indices
            edge_points = get_edge_points(mesh, mesh.edges[iedge])
            x₁ = edge_points[1].x[1]
            x₂ = edge_points[2].x[1]
            if x₁ ≤ xₚ ≤ x₂ || x₂ ≤ xₚ ≤ x₁
                return iedge
            end
        end
    else # NESW == 2 || NESW == 4
        # On East or West edge. Just check y coordinates
        yₚ = p.x[2]                                      
        for iedge in boundary_edge_indices
            edge_points = get_edge_points(mesh, mesh.edges[iedge])
            y₁ = edge_points[1].x[2]
            y₂ = edge_points[2].x[2]
            if y₁ ≤ yₚ ≤ y₂ || y₂ ≤ yₚ ≤ y₁
                return iedge
            end
        end
    end
    @error "Could not find start edge"
    return I(0)
end

# Ray trace a HRPM given the ray spacing and angular quadrature
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

# Ray trace a HRPM given the ray spacing and angular quadrature
function ray_trace(tₛ::T,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T}, 
                   mesh::UnstructuredMesh_2D{T, I}
                   ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    @info "Ray tracing"
    the_tracks = tracks(tₛ, ang_quad, mesh)
    # If the mesh has boundary edges, assume you want to use the combined segmentize
    # face index algorithm
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
    start_NESW = classify_NESW(start_point, mesh)
    end_NESW = classify_NESW(end_point, mesh)
    # Find the starting and ending edges and faces
    start_iedge = get_start_edge_NESW(start_point, mesh.boundary_edges[start_NESW], start_NESW, mesh)
    start_iface = mesh.edge_face_connectivity[start_iedge][2] # 1st entry should be 0
    end_iedge = get_start_edge_NESW(end_point, mesh.boundary_edges[end_NESW], end_NESW, mesh)
    end_iface = mesh.edge_face_connectivity[end_iedge][2] # 1st entry should be 0
    intersection_points = [start_point]
    face_indices = I[]
    if 0 < length(mesh.edges_materialized)
        ray_trace_edge_to_edge_explicit!(l, mesh, intersection_points, face_indices,
                                         start_iedge, start_iface, end_iface, 
                                         mesh.edge_face_connectivity,
                                         mesh.face_edge_connectivity,
                                         mesh.edges_materialized,
                                         mesh.faces_materialized)
    else # implicit
        ray_trace_edge_to_edge_implicit!(l, end_point, intersection_points, face_indices,
                                         iedge, iface, 
                                         mesh.edge_face_connectivity,
                                         mesh.face_edge_connectivity,
                                         mesh,
                                         mesh.edges)
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
    while !end_reached && iters < max_iters
        (iedge_next, iface_next, furthest_point) = next_edge_and_face_explicit(
                                                      start_iedge, start_iface,
                                                      l, mesh.edges_materialized,
                                                      mesh.edge_face_connectivity, 
                                                      mesh.face_edge_connectivity)
        if iface_next == iface # Or if jumping back  
            # If the next face could not be determined, or the ray is jumping back to the
            # previous face, this means either:                           
            # (1) The ray is entering or exiting through a vertex, and floating point error 
            #       means the exiting edge did not register an intersection.
            # (2) You're supremely unlucky and a fallback method kicked you to another face
            #       where the next face couldn't be determined
        end
#        if iface_previous == iface
#            # Could not find the next face to go to!
#            # This is likely due to floating point error.
#            # Try all adjacent faces
#            adjacent_faces = get_adjacent_faces(mesh, iface)
#            for face in adjacent_faces
#                fnpoints, fpoints = l ∩ mesh.faces_materialized[face] 
#                if 0 < fnpoints
#                    for point in fpoints[1:fnpoints]
#                        if distance(l.points[1], furthest_point) ≤ distance(l.points[1], point)
#                            furthest_point = point
#                            iedge = get_shared_edge(mesh, iface_previous, face) 
#                            iface = face 
#                        end
#                    end
#                end
#            end
#        end
        push!(intersection_points, furthest_point)
        push!(face_indices, iface)
        iedge_previous = iedge
        iface_previous = iface
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

function next_edge_and_face_explicit(start_iedge::I, start_iface::I, l::LineSegment_2D{T},
                                     edges_materialized::Vector{LineSegment_2D{T}},
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
        if edge_id == iedge
            continue
        end
        # Edges are linear, so 1 intersection point max
        npoints, point = l ∩ edges_materialized[edge_id]
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
            end
        end
    end
    return iedge_next, iface_next, furthest_point 
end


#function ray_trace_edge_to_edge_explicit!(l::LineSegment_2D{T},
#                                          mesh::UnstructuredMesh_2D{T, I},
#                                          end_point::Point_2D{T},
#                                          intersection_points::Vector{Point_2D{T}},
#                                          face_indices::Vector{I},
#                                          iedge::I,
#                                          iface::I,
#                                          end_iface::I,
#                                          edge_face_connectivity::Vector{NTuple{2, I}},
#                                          face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}}, 
#                                          edges_materialized::Vector{LineSegment_2D{T}},
#                                          faces_materialized::Vector{<:Face_2D{T}}
#                                          ) where {T <: AbstractFloat, I <: Unsigned}
#    max_iters = Int64(1E5)
#    iedge_previous = iedge
#    iface_previous = iface
#    end_reached = false
#    iters = 0
#    println("Start iedge: $iedge")
#    println("Start iface: $iface")
#    println("end_point: $end_point")
#    f = Figure()
#    display(f)
#    ax = Axis(f[1, 1], aspect = 1)
#    linesegments!(edges_materialized)
#    linesegments!(l)
#    while !end_reached && iters < max_iters
#        # For each edge in this face, intersect the track with the edge
#        furthest_point = l.points[1]
#        for edge_id in face_edge_connectivity[iface_previous]             
#            # If we are testing the edge we came in on, skip
#            println("iface: $iface")
#            println("edge_id: $edge_id")
#            println("furthest_point: ", furthest_point)
#            if edge_id == iedge_previous
#                println("skip")
#                continue
#            end
#            linesegments!(edges_materialized[edge_id])
#            npoints, points = l ∩ edges_materialized[edge_id]
#            # If there was an intersection, add the point
#            # Edges are linear, so only one intersection point
#            println("npoints, points: $npoints, ", points[1])
#            if 0 < npoints 
#                push!(intersection_points, points[1])
#                push!(face_indices, iface)
#                scatter!(points[1])
#                # If the point on this edge is further than the current furthest point
#                # from the start of the track, then we want to leave the face from this edge
#                if distance(l.points[1], furthest_point) ≤ distance(l.points[1], points[1])
#                    furthest_point = points[1]
#                    println("new furthest point: ", furthest_point)
#                    if edge_face_connectivity[edge_id][1] == iface_previous
#                        iface = edge_face_connectivity[edge_id][2]
#                    else
#                        iface = edge_face_connectivity[edge_id][1]
#                    end
#                    iedge = edge_id
#                    println("Now leaving this face on edge $iedge to face $iface a")
#                end
#            end
#            s = readline()
#        end
#        iters += 1
#        if iface_previous == iface
#            # Could not find the next face to go to!
#            # This is likely due to floating point error.
#            # Try all adjacent faces
#            @info "Had to use face"
#            adjacent_faces = get_adjacent_faces(mesh, iface)
#            println("adjacent_faces: ", adjacent_faces)
#            for face in adjacent_faces
#                println("face: $face")
#                fnpoints, fpoints = l ∩ mesh.faces_materialized[face] 
#                println("fnpoints, fpoints: $fnpoints, ", fpoints)
#                if 0 < fnpoints
#                    for point in fpoints[1:fnpoints]
#                        println("point: $point")
#                        if distance(l.points[1], furthest_point) ≤ distance(l.points[1], point)
#                            furthest_point = point
#                            iedge = get_shared_edge(mesh, iface_previous, face) 
#                            iface = face 
#                            println("new furthest point: ", furthest_point)
#                            println("Now leaving this face on edge $iedge to face $iface a")
#                        end
#                    end
#                end
#                s = readline()
#            end
#        end
#        iedge_previous = iedge
#        iface_previous = iface
#        println("Iteration over")
#        println("")
#        println("")
#        println("")
#        # If the furthest intersection is below the minimum segment length to the
#        # end point, end here.
#        if distance(furthest_point, end_point) < minimum_segment_length
#            end_reached = true
#            if furthest_point != end_point
#                push!(intersection_points, end_point)
#                push!(face_indices, end_iface)
#            end
#        end
#    end
#    if max_iters ≤ iters
#        @error "Exceeded max iterations for $l"
#    end
#end

function ray_trace_edge_to_edge_explicit!(l::LineSegment_2D{T},
                                          end_point::Point_2D{T},
                                          intersection_points::Vector{Point_2D{T}},
                                          face_indices::Vector{I},
                                          iedge::I,
                                          iface::I,
                                          edge_face_connectivity::Vector{NTuple{2, I}},
                                          face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}}, 
                                          edges_materialized::Vector{QuadraticSegment_2D{T}}
                                          ) where {T <: AbstractFloat, I <: Unsigned}
    max_iters = Int64(1E5)
    iedge_previous = iedge
    end_reached = false
    iters = 0
    while !end_reached && iters < max_iters
        # For each edge in this face, intersect the track with the edge
        last_point = last(intersection_points) 
        furthest_point = last_point
        for edge_id in face_edge_connectivity[iface]             
            # If we are testing the edge we came in on, skip
            if edge_id == iedge_previous
                continue
            end
            npoints, points = l ∩ edges_materialized[edge_id]
            # If there was an intersection, add the point
            # Edges are linear, so only one intersection point
            if 0 < npoints
                append!(intersection_points, [points[1]])
                push!(face_indices, I(iface))
                # If the point is further than the current furthest point
                # from the point the ray entered the face on, then we want to leave
                # the face from this edge
                if distance(last_point, furthest_point) < distance(last_point, points[1])
                    furthest_point = points[1]
                    if edge_face_connectivity[edge_id][1] == iface
                        iface = edge_face_connectivity[edge_id][2]
                    else
                        iface = edge_face_connectivity[edge_id][1]
                    end
                    iedge = edge_id
                end
            end
        end
        iters += 1
        iedge_previous = iedge
        # If the most recent intersection is below the minimum segment length to the
        # end point, end here.
        last_point = last(intersection_points)
        if distance(last_point, end_point) < minimum_segment_length
            end_reached = true
            if last_point != end_point
                push!(intersection_points, end_point)
            end
        end
    end
end

function ray_trace_edge_to_edge_implicit!(l::LineSegment_2D{T},
                                          end_point::Point_2D{T},
                                          intersection_points::Vector{Point_2D{T}},
                                          face_indices::Vector{I},
                                          iedge::I,
                                          iface::I,
                                          edge_face_connectivity::Vector{NTuple{2, I}},
                                          face_edge_connectivity::Vector{<:Union{NTuple{3, I}, 
                                                                                 NTuple{4, I}}}, 
                                          mesh::UnstructuredMesh_2D{T, I},
                                          edges::Vector{NTuple{2, I}}
                                          ) where {T <: AbstractFloat, I <: Unsigned}
    iedge_previous = iedge
    end_reached = false
    while !end_reached
       # For each edge in this face, intersect the track with the edge
        for edge_id in face_edge_connectivity[iface]             
            if edge_id == iedge_previous
                continue
            end
            edge = edges[edge_id]
            npoints, points = l ∩ LineSegment_2D(get_edge_points(mesh, edge))
            # If there was an intersection, move to the next face
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
                push!(face_indices, I(iface))
                if edge_face_connectivity[edge_id][1] == iface
                    iface = edge_face_connectivity[edge_id][2]
                else
                    iface = edge_face_connectivity[edge_id][1]
                end
                iedge = edge_id
            end
        end
        iedge_previous = iedge
        # If the most recent intersection is below the minimum segment length to the
        # end point, end here.
        last_point = last(intersection_points)
        if distance(last_point, end_point) < minimum_segment_length
            end_reached = true
            if last_point != end_point
                push!(intersection_points, end_point)
            end
        end
    end
end

function ray_trace_edge_to_edge_implicit!(l::LineSegment_2D{T},
                                          end_point::Point_2D{T},
                                          intersection_points::Vector{Point_2D{T}},
                                          face_indices::Vector{I},
                                          iedge::I,
                                          iface::I,
                                          edge_face_connectivity::Vector{NTuple{2, I}},
                                          face_edge_connectivity::Vector{<:Union{NTuple{3, I}, 
                                                                                 NTuple{4, I}}}, 
                                          mesh::UnstructuredMesh_2D{T, I},
                                          edges::Vector{NTuple{3, I}}
                                          ) where {T <: AbstractFloat, I <: Unsigned}
    iedge_previous = iedge
    end_reached = false
    while !end_reached
       # For each edge in this face, intersect the track with the edge
        for edge_id in face_edge_connectivity[iface]             
            if edge_id == iedge_previous
                continue
            end
            edge = edges[edge_id]
            npoints, points = l ∩ QuadraticSegment_2D(get_edge_points(mesh, edge))
            # If there was an intersection, move to the next face
            if 0 < npoints
                append!(intersection_points, collect(points[1:npoints]))
                push!(face_indices, I(iface))
                if edge_face_connectivity[edge_id][1] == iface
                    iface = edge_face_connectivity[edge_id][2]
                else
                    iface = edge_face_connectivity[edge_id][1]
                end
                iedge = edge_id
            end
        end
        iedge_previous = iedge
        # If the most recent intersection is below the minimum segment length to the
        # end point, end here.
        last_point = last(intersection_points)
        if distance(last_point, end_point) < minimum_segment_length
            end_reached = true
            if last_point != end_point
                push!(intersection_points, end_point)
            end
        end
    end
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
    if !are_faces_materialized(HRPM)
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
    if !(0 < length(mesh.faces_materialized))
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

function tracks(tₛ::T,
                ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T},
                mesh::UnstructuredMesh_2D{T, I};
                boundary_shape="Rectangle"
                ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}

    if boundary_shape == "Rectangle"
        bb = AABB(mesh, rectangular_boundary=true)
        w = bb.points[3].x[1] - bb.points[1].x[1]
        h = bb.points[3].x[2] - bb.points[1].x[2]
        # The tracks for each γ
        the_tracks = [ tracks(tₛ, w, h, γ) for γ in ang_quad.γ ]  
        # Shift all tracks if necessary, since the tracks are generated as if the HRPM has a 
        # bottom left corner at (0,0)
        offset = bb.points[1]
        for angle in the_tracks
            for track in angle
                track = LineSegment_2D(track.points[1] + offset, track.points[2] + offset)
            end
        end
        return the_tracks
    else
        @error "Unsupported boundary shape"
        return Vector{LineSegment_2D{T}}[]
    end
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
