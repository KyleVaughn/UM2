# Routines for extracting segment/face data for tracks (rays) overlaid on a mesh

# Classify a point as on the North, East, South, or West boundary edge of a rectangular mesh
function classify_nesw(p::Point_2D, mesh::UnstructuredMesh_2D) 
    ϵ = 1e-4 
    # First boundary edge ID -> get the first point in the edge -> Get the desired coordinate
    y_N = mesh.points[mesh.edges[mesh.boundary_edges[1][1]][1]][2]
    x_E = mesh.points[mesh.edges[mesh.boundary_edges[2][1]][1]][1]
    y_S = mesh.points[mesh.edges[mesh.boundary_edges[3][1]][1]][2]
    x_W = mesh.points[mesh.edges[mesh.boundary_edges[4][1]][1]][1]
    if abs(p[2] - y_N) < ϵ
        return 1 # North
    elseif abs(p[1] - x_E) < ϵ
        return 2 # East
    elseif abs(p[2] - y_S) < ϵ 
        return 3 # South
    elseif abs(p[1] - x_W) < ϵ 
        return 4 # West
    else
        # Used as an index to boundary_edges, so error should be evident
        return 0 # Error
    end
end

# # Get the face indices for all tracks in a single angle
# # Type-stable
# function find_segment_faces_in_angle!(segment_points::Vector{Vector{Point_2D{F}}},
#                                       indices::Vector{Vector{MVector{N, U}}},
#                                       HRPM::HierarchicalRectangularlyPartitionedMesh{F, U}
#                                      ) where {F <: AbstractFloat, U <: Unsigned, N}
#     nt = length(segment_points)
#     nfaces_found = 0
#     # for each track, find the segment indices
#     for it = 1:nt
#         # Returns true if indices were found for all segments in the track
#         nfaces_found += Int64(find_segment_faces_in_track!(segment_points[it], indices[it], HRPM))
#     end
#     return nfaces_found == nt
# end
# 
# # Get the face indices for all segments in a single track
# # Type-stable
# function find_segment_faces_in_track!(segment_points::Vector{Point_2D{F}},
#                                       indices::Vector{MVector{N, U}},
#                                       HRPM::HierarchicalRectangularlyPartitionedMesh{F, U}
#                                      ) where {F <: AbstractFloat, U <: Unsigned, N}
#     # Points in the track
#     npoints = length(segment_points)
#     nfaces_found = 0
#     # Test the midpoint of each segment to find the face
#     for iseg = 1:npoints-1
#         p_midpoint = midpoint(segment_points[iseg], segment_points[iseg+1])
#         nfaces_found += Int64(findface!(p_midpoint, indices[iseg], HRPM))
#     end
#     return nfaces_found == npoints - 1
# end
# 
# Get the face indices for all tracks in a single angle
function find_segment_faces_in_angle!(segment_points::Vector{Vector{Point_2D}},
                                      segment_faces::Vector{Vector{UInt32}},
                                      mesh::UnstructuredMesh_2D)
    # for each track, find the segment indices
    for it = 1:length(segment_points)
        # Returns true if indices were found for all segments in the track
        find_segment_faces_in_track!(segment_points[it], segment_faces[it], mesh)
    end
    return nothing 
end

# Get the face indices for all segments in a single track
function find_segment_faces_in_track!(segment_points::Vector{Point_2D},
                                      segment_faces::Vector{UInt32},
                                      mesh::UnstructuredMesh_2D)
    npoints = length(segment_points)
    # Test the midpoint of each segment to find the face
    for iseg = 1:npoints-1
        p_midpoint = midpoint(segment_points[iseg], segment_points[iseg+1])
        segment_faces[iseg] = findface(p_midpoint, mesh)
    end
    return nothing 
end

# # Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html
# # Generate tracks with track spacing tₛ for each azimuthal angle in the angular quadrature. 
# # These tracks lie within the domain of the mesh.
# # Type-stable
# function generate_tracks(tₛ::F,
#                          ang_quad::ProductAngularQuadrature{nᵧ, nₚ, F},
#                          HRPM::HierarchicalRectangularlyPartitionedMesh{F, U}
#                          ) where {nᵧ, nₚ, F <: AbstractFloat, U <: Unsigned}
#     w = HRPM_width(HRPM)
#     h = HRPM_height(HRPM)
#     # The tracks for each γ
#     tracks = [ generate_tracks(γ, tₛ, w, h) for γ in ang_quad.γ ].data
#     # Shift all tracks if necessary, since the tracks are generated as if the HRPM has a
#     # bottom left corner at (0,0)
#     offset = HRPM.rect.points[1]
#     for iγ = 1:nᵧ  
#         for it in 1:length(tracks[iγ])
#             tracks[iγ][it] = LineSegment_2D(tracks[iγ][it].points[1] + offset, 
#                                             tracks[iγ][it].points[2] + offset)
#         end
#     end
#     return tracks
# end
# 
# Generate tracks with track spacing tₛ for each azimuthal angle in the angular quadrature. 
# These tracks lie within the domain of the mesh.
# Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html
function generate_tracks(tₛ::Float64,
                         ang_quad::ProductAngularQuadrature{nᵧ, nₚ},
                         mesh::UnstructuredMesh_2D;
                         boundary_shape::String = "Unknown"
                        ) where {nᵧ, nₚ}
    if boundary_shape !== "Rectangle"
        @error "Unsupported boundary shape"
    end
    bb = boundingbox(mesh, boundary_shape = "Rectangle")
    # The tracks for each γ
    tracks = [ generate_tracks(γ, tₛ, width(bb), height(bb)) for γ in ang_quad.γ ].data
    # Shift all tracks if necessary, since the tracks are generated as if the HRPM has a
    # bottom left corner at (0,0)
    offset = bb.bl
    for iγ = 1:nᵧ  
        tracks[iγ] .+= offset 
    end
    return tracks
end

# Generate tracks for angle γ, with track spacing tₛ for a rectangular domain with width w, height h
# Rectangle has bottom left corner at (0, 0)
function generate_tracks(γ::Float64, tₛ::Float64, w::Float64, h::Float64)
    if γ ≤ π/2
        n_y = ceil(Int64, h*cos(γ)/tₛ)
        # Number of tracks in x direction
        n_x = ceil(Int64, w*sin(γ)/tₛ)
        # Total number of tracks
        nₜ = n_y + n_x
        # Allocate the tracks
        tracks = Vector{LineSegment_2D}(undef, nₜ)
        # Effective angle to ensure cyclic tracks
        γₑ = atan((h*n_x)/(w*n_y))
        t_eff = w*sin(γₑ)/n_x
        Δx = t_eff/sin(γₑ) 
        Δy = t_eff/cos(γₑ) 
        # Number of tracks in y direction
        # Generate tracks from the bottom edge of the rectangular domain
        for ix = 1:n_x
            x₀ = w - (ix - 0.5)*Δx
            y₀ = 0.0
            # Segment either terminates at the right edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = min(w, h/tan(γₑ) + x₀)
            y₁ = min((w - x₀)*tan(γₑ), h)
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arclength(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            tracks[ix] = l
        end
        # Generate tracks from the left edge of the rectangular domain
        for iy = 1:n_y
            x₀ = 0.0
            y₀ = (iy - 0.5)*Δy
            # Segment either terminates at the right edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = min(w, (h - y₀)/tan(γₑ))
            y₁ = min(w*tan(γₑ) + y₀, h)
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arclength(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            tracks[n_x + iy] = l
        end
    else
        γᶜ = π - γ # Compliment
        # Number of tracks in y direction
        n_y = ceil(Int64, h*cos(γᶜ)/tₛ)
        # Number of tracks in x direction
        n_x = ceil(Int64, w*sin(γᶜ)/tₛ)
        # Total number of tracks
        nₜ = n_y + n_x
        # Allocate the tracks
        tracks = Vector{LineSegment_2D}(undef, nₜ)
        # Effective angle to ensure cyclic tracks
        γₑ = atan((h*n_x)/(w*n_y))
        t_eff = w*sin(γₑ)/n_x
        Δx = t_eff/sin(γₑ) 
        Δy = t_eff/cos(γₑ) 
        # Generate tracks from the bottom edge of the rectangular domain
        for ix = 1:n_x
            x₀ = (ix - 0.5)*Δx
            y₀ = 0.0
            # Segment either terminates at the left edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = max(0, -h/tan(γₑ) + x₀)
            y₁ = min(x₀*tan(γₑ), h)
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arclength(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            tracks[ix] = l
        end
        # Generate tracks from the right edge of the rectangular domain
        for iy = 1:n_y
            x₀ = w
            y₀ = (iy - 0.5)*Δy
            # Segment either terminates at the left edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = max(0, w - (h - y₀)/tan(γₑ))
            y₁ = min(w*tan(γₑ) + y₀, h)
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arclength(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            tracks[n_x + iy] = l
        end
     end
    return tracks
end

# Get the boundary edge that a point lies on for a rectangular mesh
function get_start_edge_nesw(p::Point_2D, mesh::UnstructuredMesh_2D)
    nesw = classify_nesw(p, mesh)
    if nesw == 1 || nesw == 3
        # On North or South edge. Just check x coordinates
        xₚ = p[1]
        for iedge in mesh.boundary_edges[nesw]
            epoints = edgepoints(mesh.edges[iedge], mesh.points)
            x₁ = epoints[1][1]
            x₂ = epoints[2][1]
            if x₁ ≤ xₚ ≤ x₂ || x₂ ≤ xₚ ≤ x₁
                return iedge
            end
        end
    else # nesw == 2 || nesw == 4
        # On East or West edge. Just check y coordinates
        yₚ = p[2]
        for iedge in mesh.boundary_edges[nesw]
            epoints = edgepoints(mesh.edges[iedge], mesh.points)
            y₁ = epoints[1][2]
            y₂ = epoints[2][2]
            if y₁ ≤ yₚ ≤ y₂ || y₂ ≤ yₚ ≤ y₁
                return iedge
            end
        end
    end
    # Used as index to connectivity, so error should be evident
    return 0x00000000
end

# Get the segment points and the face which the segment lies in for all segments,
# in all tracks in an angle, using the edge-to-edge ray tracing method.
# Assumes a rectangular boundary
function ray_trace_angle_edge_to_edge!(tracks::Vector{LineSegment_2D},
                                       segment_points::Vector{Vector{Point_2D}},
                                       segment_faces::Vector{Vector{UInt32}},
                                       mesh::UnstructuredMesh_2D)
    for it = 1:length(tracks)
        segment_points[it], segment_faces[it] = ray_trace_track_edge_to_edge(tracks[it], mesh) 
    end
    return nothing
end

# Get the segment points and the face which the segment lies in for all segments
# in a track, using the edge-to-edge ray tracing method.
# Assumes a rectangular boundary
# Linear edges
function ray_trace_track_edge_to_edge(l::LineSegment_2D, mesh::LinearUnstructuredMesh_2D)
    # Classify line as intersecting north, east, south, or west boundary edge of the mesh
    start_point = l.points[1]; end_point = l.points[2]
    # Find the edges and faces the line starts and ends the mesh on
    start_edge = get_start_edge_nesw(start_point, mesh)
    start_face = mesh.edge_face_connectivity[start_edge][2] # 1st entry should be 0
    end_edge   = get_start_edge_nesw(end_point, mesh)
    end_face   = mesh.edge_face_connectivity[end_edge][2] # 1st entry should be 0
    # Setup for finding the segment points, faces
    segment_points = [start_point]
    segment_faces = UInt32[]
    max_iters = Int64(1E3) # Max iterations of finding intersection points before declaring an error
    current_edge = start_edge
    current_face = start_face
    next_edge = start_edge
    next_face = start_face
    intersection_point = start_point
    last_point = start_point
    end_reached = false
    iters = 0
    # Find the segment points, faces
    while !end_reached && iters < max_iters
        if visualize_ray_tracing 
            mesh!(mesh.materialized_faces[current_face], color = (:yellow, 0.2))
        end
        next_edge, next_face, intersection_point = next_edge_and_face(last_point, current_edge, 
                                                                        current_face, l, mesh)
        # Could not find next face, or jumping back to a previous face
        if next_face === current_face
            next_edge, next_face = next_edge_and_face_fallback(last_point, current_face, 
                                                               segment_faces, l, mesh)
        else
            if visualize_ray_tracing 
                mesh!(mesh.materialized_faces[current_face], color = (:green, 0.15))
                scatter!(intersection_point, color = :green)
                linesegments!(mesh.materialized_edges[next_edge], color = :green)
                println("Intersection at point $intersection_point, on face $current_face," *
                        " over edge $next_edge")
            end
            push!(segment_points, intersection_point)
            push!(segment_faces, current_face)
            last_point = intersection_point
        end
        current_edge = next_edge
        current_face = next_face
        # If the furthest intersection is below the minimum segment length to the
        # end point, end here.
        if distance(intersection_point, end_point) < minimum_segment_length ||
                current_face == end_face
            end_reached = true
            if intersection_point ≉  end_point
                push!(segment_points, end_point)
                push!(segment_faces, end_face)
            end
        end
        iters += 1
    end
    if max_iters ≤ iters
        @error "Exceeded max iterations for $l."
    end
    return sort_intersection_points_E2E(l, segment_points, segment_faces)
end

# Get the segment points and the face which the segment lies in for all segments
# in a track, using the edge-to-edge ray tracing method.
# Assumes a rectangular boundary
# Quadratic edges
function ray_trace_track_edge_to_edge(l::LineSegment_2D, mesh::QuadraticUnstructuredMesh_2D)
    # Classify line as intersecting north, east, south, or west boundary edge of the mesh
    start_point = l.points[1]; end_point = l.points[2]
    # Find the edges and faces the line starts and ends the mesh on
    start_edge = get_start_edge_nesw(start_point, mesh)
    start_face = mesh.edge_face_connectivity[start_edge][2] # 1st entry should be 0
    end_edge   = get_start_edge_nesw(end_point, mesh)
    end_face   = mesh.edge_face_connectivity[end_edge][2] # 1st entry should be 0
    # Setup for finding the segment points, faces
    segment_points = [start_point]
    segment_faces = UInt32[]
    max_iters = Int64(1E3) # Max iterations of finding the next point before declaring an error
    current_face = start_face
    next_face = start_face
    intersection_point = start_point
    last_point = start_point
    end_reached = false
    iters = 0
    # Find the segment points, faces
    while !end_reached && iters < max_iters
        if visualize_ray_tracing 
            mesh!(mesh.materialized_faces[current_face], color = (:yellow, 0.2))
        end
        next_face, intersection_point = next_face_E2E(last_point, current_face, l, mesh)
        # Could not find next face, or jumping back to a previous face
        if next_face === current_face
            next_face = next_face_fallback(last_point, current_face, segment_faces, l, mesh) 
        else
            if visualize_ray_tracing 
                mesh!(mesh.materialized_faces[current_face], color = (:green, 0.15))
                scatter!(intersection_point, color = :green)
                println("Intersection at point $intersection_point, on face $current_face")
            end
            push!(segment_points, intersection_point)
            push!(segment_faces, current_face)
            last_point = intersection_point
        end
        current_face = next_face
        # If the furthest intersection is below the minimum segment length to the
        # end point, end here.
        if distance(intersection_point, end_point) < minimum_segment_length
            end_reached = true
            if intersection_point ≉  end_point
                push!(segment_points, end_point)
                push!(segment_faces, end_face)
            end
        end
        iters += 1
    end
    if max_iters ≤ iters
        @error "Exceeded max iterations for $l."
    end
    return sort_intersection_points_E2E(l, segment_points, segment_faces)
end
 
# Return the next edge, next face, and intersection point on the next edge
# This is for linear, materialized edges
function next_edge_and_face(last_point::Point_2D, current_edge::UInt32, current_face::UInt32, 
                            l::LineSegment_2D, mesh::UnstructuredMesh_2D)
    if visualize_ray_tracing # Compile time constant. Compiler will prune branch if not visualizing
        ax = current_axis() 
    end
    next_edge = current_edge
    next_face = current_face
    start_point = l.points[1]
    min_distance = distance(start_point, last_point) + minimum_segment_length
    intersection_point = Point_2D(1e10)
    # For each edge in this face, intersect the track with the edge
    for edge in mesh.face_edge_connectivity[current_face]
        # If we used this edge to get to this face, skip it.
        if edge === current_edge
            continue
        end
        if visualize_ray_tracing 
            lplot = linesegments!(mesh.materialized_edges[edge], color = :orange)
        end
        # Edges are linear, so 1 intersection point max
        npoints, point = l ∩ mesh.materialized_edges[edge]
        # If there's an intersection
        if npoints === 0x00000001
            # If the intersection point on this edge is further along the ray than
            # the last_point, then we want to leave the face from this edge
            if min_distance < distance(start_point, point)
                intersection_point = point
                # Make sure not to pick the current face for the next face
                if mesh.edge_face_connectivity[edge][1] == current_face
                    next_face = mesh.edge_face_connectivity[edge][2]
                else
                    next_face = mesh.edge_face_connectivity[edge][1]
                end
                next_edge = edge
                break
            end
        end
        if visualize_ray_tracing 
            readline()
            delete!(ax.scene, lplot)
        end
    end
    if visualize_ray_tracing 
        readline()
    end
    return next_edge, next_face, intersection_point
end

# Return the next face and intersection point on the next edge
# This is for quadratic, materialized edges
function next_face_E2E(last_point::Point_2D, current_face::UInt32, l::LineSegment_2D,
                       mesh::QuadraticUnstructuredMesh_2D)
    if visualize_ray_tracing # Compile time constant. Prune branch if not visualizing
        ax = current_axis() 
    end
    next_face = current_face
    start_point = l[1]
    min_distance = distance(start_point, last_point) + minimum_segment_length
    intersection_point = Point_2D(1e10)
    # For each edge in this face, intersect the track with the edge
    for edge in mesh.face_edge_connectivity[current_face]
        if visualize_ray_tracing 
            lplot = linesegments!(mesh.materialized_edges[edge], color = :orange)
        end
        npoints, ipoints = l ∩ mesh.materialized_edges[edge]
        # If there's an intersection
        for i = 1:npoints
            point = ipoints[i]
            # If the intersection point on this edge is further along the ray than
            # the last_point, then we want to leave the face from this edge
            if min_distance < distance(start_point, point) < distance(start_point, intersection_point)
                intersection_point = point
                # Make sure not to pick the current face for the next face
                if mesh.edge_face_connectivity[edge][1] == current_face
                    next_face = mesh.edge_face_connectivity[edge][2]
                else
                    next_face = mesh.edge_face_connectivity[edge][1]
                end
            end
        end
        if visualize_ray_tracing 
            readline()
            delete!(ax.scene, lplot)
        end
    end
    return next_face, intersection_point
end

# Return the next face, and edge to skip, for the edge-to-edge algorithm to check in the event 
# that they could not be determined simply by checking edges of the current face. This typically 
# means there was an intersection around a vertex, and floating point error is
# causing an intersection to erroneously return 0 points intersected.
# Requires materialized faces
function next_edge_and_face_fallback(last_point::Point_2D, current_face::UInt32, 
                                     segment_faces::Vector{UInt32}, l::LineSegment_2D,
                                     mesh::UnstructuredMesh_2D)
    # If the next face could not be determined, or the ray is jumping back to the
    # previous face, this means either:
    # (1) The ray is entering or exiting through a vertex, and floating point error
    #       means the exiting edge did not register an intersection.
    # (2) You're supremely unlucky and a fallback method kicked you to another face
    #       where the next face couldn't be determined
    next_face = current_face
    start_point = l[1]
    # The closest point along l intersected in this iteration
    closest_point = Point_2D(1e10)

    # Check adjacent faces first to see if that is sufficient to solve the problem
    next_face, closest_point = adjacent_faces_fallback(last_point, current_face, l, mesh) 
    # If adjacent faces were not sufficient, try all faces sharing the vertices of this face
    if next_face === current_face || next_face ∈  segment_faces
        next_face, closest_point = shared_vertex_fallback(last_point, current_face, l, mesh) 
        if visualize_ray_tracing 
            readline()
        end
    end 

    # If the next face STILL couldn't be determined, you're screwed
    if next_face == current_face
        @error "Could not find next face, even using fallback methods, for segment $l."  
    end
    # Determine the edge that should be skipped by choosing the edge with intersection point 
    # closest to the start of the line.
    next_edge = skipped_edge_fallback(next_face, l, mesh) 
    return next_edge, next_face
end

# Return the next face for the edge-to-edge algorithm to check in the event 
# that it could not be determined simply by checking edges of the current face. This typically 
# means there was an intersection around a vertex, and floating point error is
# causing an intersection to erroneously return 0 points intersected.
# Requires materialized faces
# Quadratic edges
function next_face_fallback(last_point::Point_2D, current_face::UInt32, segment_faces::Vector{UInt32},
            l::LineSegment_2D, mesh::QuadraticUnstructuredMesh_2D)
    # If the next face could not be determined, this means either:
    # (1) The ray is entering or exiting through a vertex, and floating point error
    #       means the exiting edge did not register an intersection.
    # (2) You're supremely unlucky and a fallback method kicked you to another face
    #       where the next face couldn't be determined
    next_face = current_face
    start_point = l[1]
    # The closest point along l intersected in this iteration
    closest_point = Point_2D(1e10)
    next_face, closest_point = quadratic_shared_vertex_fallback(last_point, current_face, l, mesh)
    if visualize_ray_tracing 
        readline()
    end
    # If the next face STILL couldn't be determined, you're screwed
    if next_face === current_face
        @error "Could not find next face, even using fallback methods, for segment $l."  
    end
    return next_face
end

# Check to see if one of the adjacent faces should be the next face in edge-to-edge ray tracing
function adjacent_faces_fallback(last_point::Point_2D, current_face::UInt32, 
                                 l::LineSegment_2D, mesh::UnstructuredMesh_2D)
    next_face = current_face
    start_point = l.points[1]
    min_distance = distance(start_point, last_point) + minimum_segment_length
    intersection_point = Point_2D(1e10)
    for face in adjacent_faces(current_face, mesh) 
        npoints, ipoints = l ∩ mesh.materialized_faces[face]
        if 0 < npoints
            for point in ipoints[1:npoints]
                if min_distance < distance(start_point, point) < distance(start_point, intersection_point)
                    intersection_point = point
                    next_face = face
                end
            end
        end
    end
    return next_face, intersection_point
end

# Check to see if one of the faces sharing a vertex with the current face
# should be the next face in edge-to-edge ray tracing
function shared_vertex_fallback(last_point::Point_2D, current_face::UInt32, 
                                l::LineSegment_2D, mesh::UnstructuredMesh_2D)
    next_face = current_face
    start_point = l.points[1]
    min_distance = distance(start_point, last_point) + minimum_segment_length
    intersection_point = Point_2D(1e10)
    # Get the vertex ids for each vertex in the face
    already_tested = adjacent_faces(current_face, mesh)
    push!(already_tested, current_face)
    vertex_ids = mesh.faces[current_face]
    faces_OI = UInt32[] # faces of interest
    for vertex in vertex_ids
        candidate_faces = faces_sharing_vertex(vertex, mesh)
        for face in candidate_faces
            if face ∉ already_tested && face ∉ faces_OI
                push!(faces_OI, face)
            end
        end
    end
    if visualize_ray_tracing 
        mesh_vec = []
    end
    for face in faces_OI
        npoints, ipoints = l ∩ mesh.materialized_faces[face]
        if visualize_ray_tracing 
            push!(mesh_vec, mesh!(mesh.materialized_faces[face], color = (:black, 0.2)))
            readline()
        end
        if 1 < npoints
            contains_last_point = false
            for point in ipoints[1:npoints]
                if distance(point, last_point) < minimum_segment_length
                    contains_last_point = true
                end
            end
            for point in ipoints[1:npoints]
                distance_to_point = distance(start_point, point)
                # If this is a valid intersection point
                if min_distance < distance_to_point 
                    # If this point is the closest point, use this
                    if distance_to_point ⪉  distance(start_point, intersection_point) 
                        intersection_point = point
                        next_face = face
                    # If this face contains the last point, we want to prioritize this face
                    elseif contains_last_point
                        next_face = face
                    end
                end
                if visualize_ray_tracing 
                    scatter!(point, color = :yellow)
                    readline()
                end
            end
        end
    end

    if visualize_ray_tracing 
        ax = current_axis() 
        for m in mesh_vec
            delete!(ax.scene, m)
        end
    end
    return next_face, intersection_point
end

# Check to see if one of the faces sharing a vertex with the current face
# should be the next face in edge-to-edge ray tracing
function quadratic_shared_vertex_fallback(last_point::Point_2D, current_face::UInt32, 
                                          l::LineSegment_2D, mesh::QuadraticUnstructuredMesh_2D)
    next_face = current_face
    start_point = l[1]
    min_distance = distance(start_point, last_point) + minimum_segment_length
    intersection_point = Point_2D(1e10)
    # Get the vertex ids for each vertex in the face
    vertex_ids = mesh.faces[current_face]
    faces_OI = UInt32[] # faces of interest
    for vertex in vertex_ids
        candidate_faces = faces_sharing_vertex(vertex, mesh)
        for face in candidate_faces
            if face !== current_face && face ∉ faces_OI
                push!(faces_OI, face)
            end
        end
    end
    if visualize_ray_tracing 
        mesh_vec = []
    end
    next_face_contains_last_point = false
    for face in faces_OI
        npoints, ipoints = l ∩ mesh.materialized_faces[face]
        if visualize_ray_tracing 
            println("Face: $face")
            push!(mesh_vec, mesh!(mesh.materialized_faces[face], color = (:black, 0.2)))
            readline()
        end
        if 1 < npoints
            contains_last_point = false
            for point in ipoints[1:npoints]
                if distance(point, last_point) < minimum_segment_length
                    contains_last_point = true
                end
            end
            if visualize_ray_tracing 
                println(contains_last_point ? "Contains last point" : "Does not contain last point")
            end
            for point in ipoints[1:npoints]
                distance_to_point = distance(start_point, point)
                # If this is a valid intersection point
                if min_distance < distance_to_point 
                    # If this point is the closest point, use this
                    # Could be a problem for Float32.
                    if distance_to_point ⪉ distance(start_point, intersection_point) 
                        if visualize_ray_tracing                        
                            println("New intersection point")
                            println("New face: $face")
                        end
                        intersection_point = point
                        next_face = face
                        if contains_last_point
                            next_face_contains_last_point = true
                        end
                    # If this face contains the last point, we want to prioritize this face,
                    # but only if the face with the closest point doesn't also contain
                    # the last point
                    elseif contains_last_point && !next_face_contains_last_point
                        next_face = face
                        if visualize_ray_tracing                        
                            println("New face: $face")
                        end
                    end
                end
                if visualize_ray_tracing 
                    println("Point: $point")
                    scatter!(point, color = :yellow)
                    readline()
                end
            end
        end
    end

    if visualize_ray_tracing 
        ax = current_axis() 
        for m in mesh_vec
            delete!(ax.scene, m)
        end
        println("Next face: $next_face, Intersection: $intersection_point")
    end
    return next_face, intersection_point
end

# Determine the edge that should be skipped by choosing the edge with intersection point 
# closest to the start of the line.
# Linear edges
function skipped_edge_fallback(next_face::UInt32, l::LineSegment_2D, mesh::LinearUnstructuredMesh_2D)
    start_point = l.points[1]
    closest_point = Point_2D(1.0e10)
    edges_OI = mesh.face_edge_connectivity[next_face]
    next_edge = 0x00000000
    for iedge in edges_OI
        npoints, point = l ∩ mesh.materialized_edges[iedge]
        if 0 < npoints                                                                         
            if  distance(start_point, point) < distance(start_point, closest_point) 
                closest_point = point
                next_edge = iedge
            end
        end
    end
    return next_edge
end

function sort_intersection_points_E2E(l::LineSegment_2D, segment_points::Vector{Point_2D},
                                      segment_faces::Vector{UInt32})
    if 2 <= length(segment_points)
        npoints = length(segment_points)
        perm = sortperm(distance².(l[1], view(segment_points, 2:npoints)))
        permute!(view(segment_points, 2:npoints), perm)
        permute!(segment_faces, perm)
        # Eliminate any points and faces for which the distance between consecutive points
        # is less than the minimum segment length
        delete_ids = Int64[]
        id_start = 1
        for id_stop ∈ 2:npoints
            if distance²(segment_points[id_start], segment_points[id_stop]) < minimum_segment_length^2
                push!(delete_ids, id_stop)
            else
                id_start = id_stop
            end
        end
        deleteat!(segment_points, delete_ids)
        deleteat!(segment_faces, delete_ids .- 1)
        return segment_points, segment_faces
    else
        return segment_points, segment_faces
    end
end

function use_E2E_raytracing(mesh::UnstructuredMesh_2D)
    return 0 < length(mesh.boundary_edges)
end
