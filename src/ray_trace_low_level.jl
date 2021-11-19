# Routines for extracting segment/face data for tracks (rays) overlaid on a mesh

# Classify a point as on the North, East, South, or West boundary edge of a rectangular mesh
function classify_nesw(p::Point_2D{T},
                       mesh::UnstructuredMesh_2D{T, I}; 
                       ϵ::Float64 = Point_2D_differentiation_distance
                ) where {T <: AbstractFloat, I <: Unsigned}
    y_N = mesh.points[mesh.edges[mesh.boundary_edges[1][1]][1]].x[2]
    x_E = mesh.points[mesh.edges[mesh.boundary_edges[2][1]][1]].x[1]
    y_S = mesh.points[mesh.edges[mesh.boundary_edges[3][1]][1]].x[2]
    x_W = mesh.points[mesh.edges[mesh.boundary_edges[4][1]][1]].x[1]
    if abs(p.x[2] - y_N) < ϵ
        return 1 # North
    elseif abs(p.x[1] - x_E) < ϵ
        return 2 # East
    elseif abs(p.x[2] - y_S) < ϵ 
        return 3 # South
    elseif abs(p.x[1] - x_W) < ϵ 
        return 4 # West
    else
        @error "Could not classify point"
        return 0 # Error
    end
end

# Get the boundary edge that a point lies on for a rectangular mesh
function get_start_edge_nesw(p::Point_2D{T},
                             boundary_edge_indices::Vector{I},
                             nesw::Int64,
                             mesh::UnstructuredMesh_2D{T, I}
                             ) where {T <: AbstractFloat, I <: Unsigned}
    if nesw == 1 || nesw == 3
        # On North or South edge. Just check x coordinates
        xₚ = p.x[1]
        for iedge in boundary_edge_indices
            epoints = edge_points(mesh, mesh.edges[iedge])
            x₁ = epoints[1].x[1]
            x₂ = epoints[2].x[1]
            if x₁ ≤ xₚ ≤ x₂ || x₂ ≤ xₚ ≤ x₁
                return iedge
            end
        end
    else # nesw == 2 || nesw == 4
        # On East or West edge. Just check y coordinates
        yₚ = p.x[2]
        for iedge in boundary_edge_indices
            epoints = edge_points(mesh, mesh.edges[iedge])
            y₁ = epoints[1].x[2]
            y₂ = epoints[2].x[2]
            if y₁ ≤ yₚ ≤ y₂ || y₂ ≤ yₚ ≤ y₁
                return iedge
            end
        end
    end
    @error "Could not find start edge"
    return I(0)
end

# Get the segment points and the face which the segment lies in for all segments,
# in all tracks in an angle, using the edge-to-edge ray tracing method.
# Assumes a rectangular boundary
function ray_trace_angle_edge_to_edge!(tracks::Vector{LineSegment_2D{T}},
                                       segment_points::Vector{Vector{Point_2D{T}}},
                                       segment_faces::Vector{Vector{I}},
                                       mesh::UnstructuredMesh_2D{T, I}
                                       ) where {T <: AbstractFloat, I <: Unsigned}
    has_quadratic_edges = mesh.edges isa Vector{NTuple{3, I}}
    # For each track, get the segment points and segment faces
    if has_quadratic_edges
        for it = 1:length(tracks)
            (segment_points[it], 
             segment_faces[it]) = ray_trace_track_edge_to_edge_quadratic(tracks[it],
                                                                         mesh)
        end
    else
        for it = 1:length(tracks)
            (segment_points[it], 
             segment_faces[it]) = ray_trace_track_edge_to_edge_linear(tracks[it],
                                                                      mesh)
        end
    end
end

# Get the segment points and the face which the segment lies in for all segments
# in a track, using the edge-to-edge ray tracing method.
# Assumes a rectangular boundary
function ray_trace_track_edge_to_edge_linear(l::LineSegment_2D{T},
                                             mesh::UnstructuredMesh_2D{T, I}
                                            ) where {T <: AbstractFloat, I <: Unsigned}
    # Classify line as intersecting north, east, south, or west boundary edge of the mesh
    start_point = l.points[1] # line start point
    end_point = l.points[2] # line end point
    start_point_nesw = classify_nesw(start_point, mesh) # start point is on N,S,E, or W edge
    end_point_nesw = classify_nesw(end_point, mesh) # end point is on N,S,E, or W edge

    # Find the edges and faces the line starts and ends the mesh on
    start_edge = get_start_edge_nesw(start_point, mesh.boundary_edges[start_point_nesw],
                                     start_point_nesw, mesh)
    start_face = mesh.edge_face_connectivity[start_edge][2] # 1st entry should be 0
    end_edge = get_start_edge_nesw(end_point, mesh.boundary_edges[end_point_nesw],
                                   end_point_nesw, mesh)
    end_face = mesh.edge_face_connectivity[end_edge][2] # 1st entry should be 0
    # Setup for finding the segment points, faces
    segment_points = [start_point]
    segment_faces = I[]
    max_iters = Int64(1E3) # Max iterations of finding the next point before declaring an error
    current_edge = start_edge
    current_face = start_face
    next_edge = start_edge
    next_face = start_face
    intersection_point = start_point
    end_reached = false
    iters = 0
    # Find the segment points, faces
    # Since Julia uses JIT, are has_mat_edges and has_hat_faces provided at compile time?
    # Will the compiler remove dead branches through constant propagation?
    while !end_reached && iters < max_iters
        (next_edge, 
         next_face, 
         intersection_point) = next_edge_and_face_linear_explicit(
                                    current_edge, current_face, l,
                                    mesh.materialized_edges::Vector{LineSegment_2D{T}},
                                    mesh.edge_face_connectivity,
                                    mesh.face_edge_connectivity)
        # Could not find next face, or jumping back to a previous face
        if next_face == current_face || next_face ∈  segment_faces
            next_edge, next_face = next_edge_and_face_fallback_linear(current_face, 
                                                                      segment_faces, l, mesh,
                                                                      mesh.materialized_faces)
        else
            push!(segment_points, intersection_point)
            push!(segment_faces, current_face)
        end
        current_edge = next_edge
        current_face = next_face
        # If the furthest intersection is below the minimum segment length to the
        # end point, end here.
        if distance(intersection_point, end_point) < minimum_segment_length
            end_reached = true
            if intersection_point != end_point
                push!(segment_points, end_point)
                push!(segment_faces, end_face)
            end
        end
        iters += 1
    end
    if max_iters ≤ iters
        @warn "Exceeded max iterations for $l. Reverted to intersecting each edge."
        # Do it the old fashioned way
        segment_points = intersect_edges_implicit(l, mesh, mesh.edges) 
        segment_faces = [I(0) for i = 1:length(segment_points) - 1]
        find_segment_faces_in_track!(segment_points, segment_faces, mesh)
    end
    # The segment points should already be sorted. We will eliminate any points and faces
    # for which the distance between consecutive points is less than the minimum segment length
    if 2 < length(segment_points)
        # Remove duplicate points
        segment_points_reduced = [start_point]
        segment_faces_reduced = I[]
        npoints = length(segment_points)
        for i = 2:npoints
            # If the segment would be shorter than the minimum segment length, remove it.
            if minimum_segment_length < distance(last(segment_points_reduced), segment_points[i])
                push!(segment_points_reduced, segment_points[i])
                push!(segment_faces_reduced, segment_faces[i-1])
            end
        end
        return (segment_points_reduced, segment_faces_reduced)
    else
        return (segment_points, segment_faces)
    end
end

# Return the next edge, next face, and intersection point on the next edge
# This is for linear, materialized edges
function next_edge_and_face_linear_explicit(current_edge::I, current_face::I, 
                                     l::LineSegment_2D{T},
                                     materialized_edges::Vector{LineSegment_2D{T}},
                                     edge_face_connectivity::Vector{NTuple{2, I}},
                                     face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}}
                                    ) where {T <: AbstractFloat, I <: Unsigned}
    next_edge = current_edge
    next_face = current_face
    start_point = l.points[1]
    # The furthest point along l intersected in this iteration
    furthest_point = start_point
    # For each edge in this face, intersect the track with the edge
    for edge in face_edge_connectivity[current_face]
        # If we used this edge to get to this face, skip it.
        if edge == current_edge
            continue
        end
        # Edges are linear, so 1 intersection point max
        npoints, point = l ∩ materialized_edges[edge]
        # If there's an intersection
        if 0 < npoints
            # If the intersection point on this edge is further along the ray than the current
            # furthest point, then we want to leave the face from this edge
            if distance(start_point, furthest_point) ≤ distance(start_point, point) &&
                    (point ≉ start_point)
                furthest_point = point
                # Make sure not to pick the current face for the next face
                if edge_face_connectivity[edge][1] == current_face
                    next_face = edge_face_connectivity[edge][2]
                else
                    next_face = edge_face_connectivity[edge][1]
                end
                next_edge = edge
            end
        end
    end
    return next_edge, next_face, furthest_point
end

# Return the next edge, next face, and intersection point on the next edge
# This is for quadratic, materialized edges
#function next_edge_and_face_explicit(current_edge::I, current_face::I, l::LineSegment_2D{T},
#                                     materialized_edges::Vector{QuadraticSegment_2D{T}},
#                                     edge_face_connectivity::Vector{NTuple{2, I}},
#                                     face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}}
#                                    ) where {T <: AbstractFloat, I <: Unsigned}
#    next_edge = current_edge
#    next_face = current_face
#    start_point = l.points[1]
#    # The closest point to the start of l intersected in this iteration
#    closest_point = Point_2D(T, 1.0e20, 1.0e20)
#    # For each edge in this face, intersect the track with the edge
#    for edge in face_edge_connectivity[current_face]
#        # If we used this edge to get to this face, skip it.
#        if edge == current_edge
#            continue
#        end
#        # Edges are quadratic, so 2 intersection points to deal with
#        npoints, points = l ∩ materialized_edges[edge]
##        println("npoints, points: $npoints, $points")
##        linesegments!(materialized_edges[edge], color = :yellow)
#        # If there's an intersection
#        if 0 < npoints
#            for point in points[1:npoints]
##                scatter!(point)
##                println("distances ", distance(start_point, point)," ", distance(start_point, closest_point))
#                # If the intersection point on this edge is closer to the start of the ray than 
#                # the current closest point, then we want to leave the face from this edge
#                if (distance(start_point, point) ≤ distance(start_point, closest_point)) && 
#                        (point ≉ start_point)
#                    closest_point = point
##                    println("New closest point: $closest_point")
#                    # Make sure not to pick the current face for the next face
#                    if edge_face_connectivity[edge][1] == current_face
#                        next_face = edge_face_connectivity[edge][2]
#                    else
#                        next_face = edge_face_connectivity[edge][1]
#                    end
#                    next_edge = edge
#                end
#            end
#        end
##        s = readline()
#    end
#    return next_edge, next_face, closest_point
#end

# Return the next face, and edge to skip, for the edge-to-edge algorithm to check in the event 
# that they could not be determined simply by checking edges of the current face. This typically 
# means there was an intersection around a vertex, and floating point error is
# causing an intersection to erroneously return 0 points intersected.
#
# Requires materialized faces
function next_edge_and_face_fallback_linear(current_face::I, 
                                            segment_faces::Vector{I},
                                            l::LineSegment_2D{T},
                                            mesh::UnstructuredMesh_2D{T, I},
                                            materialized_faces::Vector{<:Face_2D{T}}
                                           ) where {T <: AbstractFloat, I <: Unsigned}
    # If the next face could not be determined, or the ray is jumping back to the
    # previous face, this means either:
    # (1) The ray is entering or exiting through a vertex, and floating point error
    #       means the exiting edge did not register an intersection.
    # (2) You're supremely unlucky and a fallback method kicked you to another face
    #       where the next face couldn't be determined
    next_face = current_face
    start_point = l.points[1]
    # The furthest point along l intersected in this iteration
    furthest_point = start_point
    # Check adjacent faces first to see if that is sufficient to solve the problem
    the_adjacent_faces = adjacent_faces(current_face, mesh)
    for face in the_adjacent_faces
        npoints, ipoints = l ∩ materialized_faces[face]
        if 0 < npoints
            for point in ipoints[1:npoints]
                if distance(start_point, furthest_point) ≤ distance(start_point, point) &&
                        (point ≉ start_point)
                    furthest_point = point
                    next_face = face
                end
            end
        end
    end
    # If adjacent faces were not sufficient, try all faces sharing the vertices of this face
    if next_face == current_face || next_face ∈  segment_faces
        # Get the vertex ids for each vertex in the face
        npoints = length(mesh.faces[current_face])
        points = mesh.faces[current_face][2:npoints]
        faces = Set{Int64}()
        for point in points
            union!(faces, faces_sharing_vertex(point, mesh))
        end
        for face in faces
            npoints, ipoints = l ∩ materialized_faces[face]
            if 0 < npoints
                for point in ipoints[1:npoints]
                    if distance(start_point, furthest_point) ≤ distance(start_point, point) &&
                            (point ≉ start_point)
                        furthest_point = point
                        next_face = face
                    end
                end
            end
        end
    end 
    # If faces sharing this face's vertices was not enough, try the faces sharing a vertex approach 
    # above, but expand to the vertices of the faces sharing vertices of the current face
    if next_face == current_face || next_face ∈  segment_faces
        # Get the vertex ids for each vertex in the face
        npoints = length(mesh.faces[current_face])
        points = mesh.faces[current_face][2:npoints]
        faces_level1 = Set{Int64}()
        for point in points
            union!(faces_level1, faces_sharing_vertex(point, mesh))
        end
        faces = Set{Int64}()
        for face in faces_level1
            npoints = length(mesh.faces[face])
            points = mesh.faces[face][2:npoints]
            for point in points
                union!(faces, faces_sharing_vertex(point, mesh))
            end
        end
        for face in faces
            npoints, ipoints = l ∩ materialized_faces[face]
            if 0 < npoints
                for point in ipoints[1:npoints]
                    if distance(start_point, furthest_point) ≤ distance(start_point, point) &&
                            (point ≉ start_point)
                        furthest_point = point
                        next_face = face
                    end
                end
            end
        end
    end
    # If the next face STILL couldn't be determined, you're screwed
    if next_face == current_face || next_face ∈  segment_faces
        @error "Could not find next face, even using fallback methods."  
        println("Error segment: $l")
        println("Current_face: $current_face")
    end
    # Determine the edge that should be skipped by choosing the edge with intersection point 
    # closest to the start of the line.
    closest_point = Point_2D(T, 1.0e20, 1.0e20)
    edges = mesh.face_edge_connectivity[next_face]
    next_edge = I(0)
    for iedge in edges
        edge::LineSegment_2D = mesh.materialized_edges[iedge]
        npoints, point = l ∩ edge
        if 0 < npoints                                                                         
            if  distance(start_point, point) ≤ distance(start_point, closest_point) 
                closest_point = point
                next_edge = iedge
            end
        end
    end
    return I(next_edge), I(next_face)
end

# Return the next face the edge-to-edge algorithm should check in the event that it
# could not be determined simply by checking edges of the current face. This typically 
# means there was an intersection around a vertex, and floating point error is
# causing an intersection to erroneously return 0 points intersected.
#
# Cannot have materialized faces

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

# Get the face indices for all tracks in a single angle
function find_segment_faces_in_angle!(segment_points::Vector{Vector{Point_2D{T}}},
                                      segment_faces::Vector{Vector{I}},
                                      mesh::UnstructuredMesh_2D{T, I}
                                     ) where {T <: AbstractFloat, I <: Unsigned, N}
    nt = length(segment_points)
    bools = fill(false, nt)
    # for each track, find the segment indices
    for it = 1:nt
        # Points in the track
        npoints = length(segment_points[it])
        # Returns true if indices were found for all segments in the track
        bools[it] = find_segment_faces_in_track!(segment_points[it], segment_faces[it], mesh)
    end
    return all(bools)
end

# Get the face indices for all segments in a single track
function find_segment_faces_in_track!(segment_points::Vector{Point_2D{T}},
                                      segment_faces::Vector{I},
                                      mesh::UnstructuredMesh_2D{T, I}
                                     ) where {T <: AbstractFloat, I <: Unsigned, N}
    # Points in the track
    npoints = length(segment_points)
    bools = fill(false, npoints-1)
    # Test the midpoint of each segment to find the face
    for iseg = 1:npoints-1
        p_midpoint = midpoint(segment_points[iseg], segment_points[iseg+1])
        segment_faces[iseg] = find_face(p_midpoint, mesh)
        bools[iseg] = 0 < segment_faces[iseg]
    end
    return all(bools)
end

# Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html
function generate_tracks(tₛ::T,
                         ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T},
                         HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                         ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    w = width(HRPM)
    h = height(HRPM)
    # The tracks for each γ
    tracks = [ generate_tracks(tₛ, w, h, γ) for γ in ang_quad.γ ]
    # Shift all tracks if necessary, since the tracks are generated as if the HRPM has a
    # bottom left corner at (0,0)
    offset = HRPM.rect.points[1]
    for angle in tracks
        for track in angle
            track = LineSegment_2D(track.points[1] + offset, track.points[2] + offset)
        end
    end
    return tracks
end

function generate_tracks(tₛ::T,
                         ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T},
                         mesh::UnstructuredMesh_2D{T, I};
                         boundary_shape::String = "Unknown"
                         ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}

    if boundary_shape == "Rectangle"
        bb = bounding_box(mesh, rectangular_boundary=true)
        w = bb.points[3].x[1] - bb.points[1].x[1]
        h = bb.points[3].x[2] - bb.points[1].x[2]
        # The tracks for each γ
        tracks = [ generate_tracks(tₛ, w, h, γ) for γ in ang_quad.γ ]
        # Shift all tracks if necessary, since the tracks are generated as if the HRPM has a
        # bottom left corner at (0,0)
        offset = bb.points[1]
        for angle in tracks
            for track in angle
                track = LineSegment_2D(track.points[1] + offset, track.points[2] + offset)
            end
        end
        return tracks
    else
        @error "Unsupported boundary shape"
        return Vector{LineSegment_2D{T}}[]
    end
end

function generate_tracks(tₛ::T, w::T, h::T, γ::T) where {T <: AbstractFloat}
    # Number of tracks in y direction
    n_y = ceil(Int64, w*abs(sin(γ))/tₛ)
    # Number of tracks in x direction
    n_x = ceil(Int64, h*abs(cos(γ))/tₛ)
    # Total number of tracks
    nₜ = n_y + n_x
    # Allocate the tracks
    tracks = Vector{LineSegment_2D{T}}(undef, nₜ)
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
            tracks[ix] = l
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
            tracks[n_x + iy] = l
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
            tracks[ix] = l
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
            tracks[n_y + iy] = l
        end
    end
    return tracks
end
