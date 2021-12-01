# Return a vector of the faces adjacent to the face of ID face
# Type-stable if all of the faces/edges are the same
function adjacent_faces(face::U,
                        face_edge_connectivity::Vector{<:SVector{L, U} where {L}},
                        edge_face_connectivity::Vector{SVector{2, U}}
                       ) where {U <: Unsigned}
    edges = face_edge_connectivity[face]
    the_adjacent_faces = U[]
    for edge in edges
        faces = edge_face_connectivity[edge]
        for face_id in faces
            if face_id != face && face_id != 0
                push!(the_adjacent_faces, face_id)
            end
        end
    end
    return the_adjacent_faces
end

# Area of face
# Type-stable
function area(face::SVector{N, U}, 
              points::Vector{Point_2D{F}}) where {N, F <: AbstractFloat, U <: Unsigned}
    return area(materialize_face(face, points))
end

# Return the area of a face set
# Type-stable
function area(materialized_faces::Vector{<:Face_2D{F}},
              face_set::Set{U}) where {F <: AbstractFloat, U <: Unsigned}
    return mapreduce(x->area(materialized_faces[x]), +, face_set)
end

# Return the area of a face set
# Type-stable
function area(faces::Vector{<:SVector{L, U} where {L}},
              points::Vector{Point_2D{F}}, face_set::Set{U}) where {F <: AbstractFloat, U <: Unsigned}
    return mapreduce(x->area(faces[x], points), +, face_set)
end

# Return a vector containing vectors of the edges in each side of the mesh's bounding shape, e.g.
# For a rectangular bounding shape the sides are North, East, South, West. Then the output would
# be [ [e1, e2, e3, ...], [e17, e18, e18, ...], ..., [e100, e101, ...]]
# Not type-stable
function boundary_edges(mesh::UnstructuredMesh_2D{F, U}, 
                        bounding_shape::String) where {F <: AbstractFloat, U <: Unsigned}
    # edges which have face 0 in their edge_face connectivity are boundary edges
    the_boundary_edges = findall(x->x[1] == 0, mesh.edge_face_connectivity)
    if bounding_shape == "Rectangle"
        # Sort edges into NESW
        bb = bounding_box(mesh, rectangular_boundary=true)
        y_north = bb.points[3][2]
        x_east  = bb.points[3][1]
        y_south = bb.points[1][2]
        x_west  = bb.points[1][1]
        p_NW = bb.points[4]
        p_NE = bb.points[3]
        p_SE = bb.points[2]
        p_SW = bb.points[1]
        edges_north = U[]
        edges_east = U[]
        edges_south = U[]
        edges_west = U[]
        # Unsert edges so that indices move from NW -> NE -> SE -> SW -> NW
        for i = 1:length(the_boundary_edges)
            edge = U(the_boundary_edges[i])
            epoints = edge_points(mesh.edges[edge], mesh.points)
            if all(x->abs(x[2] - y_north) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_north, mesh.edges, p_NW, mesh.points)
            elseif all(x->abs(x[1] - x_east) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_east, mesh.edges, p_NE, mesh.points)
            elseif all(x->abs(x[2] - y_south) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_south, mesh.edges, p_SE, mesh.points)
            elseif all(x->abs(x[1] - x_west) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_west, mesh.edges, p_SW, mesh.points)
            else
                @error "Edge $iedge could not be classified as NSEW"
            end
        end
        return [ edges_north, edges_east, edges_south, edges_west ]
    else
        return [ convert(Vector{U}, the_boundary_edges) ]
    end
end

# SVector of MVectors of point IDs representing the 3 edges of a triangle
# Type-stable
function edges(face::SVector{4, U}) where {U <: Unsigned}
    edges = SVector( MVector{2, U}(face[2], face[3]),
                     MVector{2, U}(face[3], face[4]),
                     MVector{2, U}(face[4], face[2]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 4 edges of a quadrilateral
# Type-stable
function edges(face::SVector{5, U}) where {U <: Unsigned}
    edges = SVector( MVector{2, U}(face[2], face[3]),
                     MVector{2, U}(face[3], face[4]),
                     MVector{2, U}(face[4], face[5]),
                     MVector{2, U}(face[5], face[2]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 3 edges of a quadratic triangle
# Type-stable
function edges(face::SVector{7, U}) where {U <: Unsigned}
    edges = SVector( MVector{3, U}(face[2], face[3], face[5]),
                     MVector{3, U}(face[3], face[4], face[6]),
                     MVector{3, U}(face[4], face[2], face[7]) )
    # Order the linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# SVector of MVectors of point IDs representing the 4 edges of a quadratic quadrilateral
# Type-stable
function edges(face::SVector{9, U}) where {U <: Unsigned}
    edges = SVector( MVector{3, U}(face[2], face[3], face[6]),
                     MVector{3, U}(face[3], face[4], face[7]),
                     MVector{3, U}(face[4], face[5], face[8]),
                     MVector{3, U}(face[5], face[2], face[9]) )
    # Order th linear edge vertices by ID
    for edge in edges
        if edge[2] < edge[1]
            e1 = edge[1]
            edge[1] = edge[2]
            edge[2] = e1
        end
    end
    return edges
end

# The unique edges from a vector of triangles or quadrilaterals represented by point IDs
# Type-stable if faces are the same type
function edges(faces::Vector{<:Union{SVector{N, U}}}) where {N, U <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(unique(edges_unfiltered))
    return [ SVector(e.data) for e in edges_filtered ]
end

# A vector of length 2 SVectors, denoting the face ID each edge is connected to. If the edge
# is a boundary edge, face ID 0 is returned
# Type-stable, other than the error messages.
function edge_face_connectivity(the_edges::Vector{<:SVector{L, U} where {L}},
                                the_faces::Vector{<:SVector{L, U} where {L}},
                                the_face_edge_connectivity::Vector{<:SVector{L, U} where {L}}
                               ) where {U <: Unsigned}
    # Each edge should only border 2 faces if it is an interior edge, and 1 face if it is
    # a boundary edge.
    # Loop through each face in the face_edge_connectivity vector and mark each edge with
    # the faces that it borders.
    if length(the_edges) === 0
        @error "Does not have edges!"
    end
    if length(the_face_edge_connectivity) === 0
        @error "Does not have face/edge connectivity!"
    end
    edge_face = [MVector{2, U}(zeros(U, 2)) for i in eachindex(the_edges)]
    for (iface, edges) in enumerate(the_face_edge_connectivity)
        for iedge in edges
            # Add the face id in the first non-zero position of the edge_face conn. vec.
            if edge_face[iedge][1] == 0
                edge_face[iedge][1] = iface
            elseif edge_face[iedge][2] == 0
                edge_face[iedge][2] = iface
            else
                @error "Edge $iedge seems to have 3 faces associated with it!"
            end
        end
    end
    return [SVector(sort(two_faces).data) for two_faces in edge_face]
end

# Return an SVector of the points in the edge
# Type-stable
function edge_points(edge::SVector{2, U},
                     points::Vector{Point_2D{F}}
                    ) where {F <: AbstractFloat, U <: Unsigned}
    return SVector(points[edge[1]], points[edge[2]])
end

# Return an SVector of the points in the edge
# Type-stable
function edge_points(edge::SVector{3, U},
                     points::Vector{Point_2D{F}}
                    ) where {F <: AbstractFloat, U <: Unsigned}
    return SVector(points[edge[1]],
                   points[edge[2]],
                   points[edge[3]]
                  )
end

# A vector of SVectors, denoting the edge ID each face is connected to.
# Not type-stable
function face_edge_connectivity(the_faces::Vector{<:SVector{L, U} where {L}},
                                the_edges::Vector{<:SVector{L, U} where {L}}
                               ) where {U <: Unsigned}
    if length(the_edges) === 0
        @error "Does not have edges!"
    end
    # A vector of MVectors of zeros for each face
    # Each MVector is the length of the number of edges
    face_edge = [MVector{Int64(num_edges(face)), U}(zeros(U, num_edges(face)))
                    for face in the_faces]::Vector{<:MVector{L, U} where {L}}
    # for each face in the mesh, generate the edges.
    # Search for the index of the edge in the mesh.edges vector
    # Insert the index of the edge into the face_edge connectivity vector
    for i in eachindex(the_faces)
        for (j, edge) in enumerate(edges(the_faces[i]))
            face_edge[i][j] = searchsortedfirst(the_edges, SVector(edge.data))
        end
    end
    return [SVector(sort(conn).data) for conn in face_edge]::Vector{<:SVector{L, U} where {L}}
end

# Return an SVector of the points in the face
# Type-stable
function face_points(face::SVector{N, U}, points::Vector{Point_2D{F}}
    ) where {N, F <: AbstractFloat, U <: Unsigned}
    return SVector{N-1, Point_2D{F}}(points[face[2:N]])
end

# Find the faces which share the vertex of ID v.
# Type-stable
function faces_sharing_vertex(v::I, faces::Vector{<:SVector{L, U} where L}) where {I <: Integer,
                                                                                   U <: Unsigned}
    shared_faces = U[]
    for i = 1:length(faces)
        N = length(faces[i])
        if v ∈  faces[i][2:N]
            push!(shared_faces, U(i))
        end
    end
    return shared_faces
end

# Find the face containing the point p, with explicitly represented faces
# Type-stable
function find_face_explicit(p::Point_2D{F},
                            faces::Vector{<:Face_2D{F}}
                           ) where {F <: AbstractFloat}
    for i = 1:length(faces)
        if p ∈  faces[i]
            return i
        end
    end
    return 0
end

# Return the face containing the point p, with implicitly represented faces
# Type-stable
function find_face_implicit(p::Point_2D{F},
                            faces::Vector{<:SVector{N, U} where N},
                            points::Vector{Point_2D{F}}
                            ) where {F <: AbstractFloat, U <: Unsigned}
    for i = 1:length(faces)
        bool = p ∈  materialize_face(faces[i], points)
        if bool
            return i
        end
    end
    return 0
end

# Insert the boundary edge into the correct place in the vector of edge indices, based on
# the distance from some reference point
# Type-stable
function insert_boundary_edge!(edge_index::U, edge_indices::Vector{U},
                               edges::Vector{<:SVector{L, U} where {L}},
                               p_ref::Point_2D{F}, points::Vector{Point_2D{F}}
                              ) where {F <: AbstractFloat, U <: Unsigned}
    # Compute the minimum distance from the edge to be inserted to the reference point
    epoints = edge_points(edges[edge_index], points)
    insertion_distance = minimum([ distance(p_ref, p_edge) for p_edge in epoints ])
    # Loop through the edge indices until an edge with greater distance from the reference point
    # is found, then insert
    nindices = length(edge_indices)
    for i = 1:nindices
        edge = edge_indices[i]
        epoints = edge_points(edges[edge], points)
        edge_distance = minimum([ distance(p_ref, p_edge) for p_edge in epoints ])
        if insertion_distance < edge_distance
            insert!(edge_indices, i, edge_index)
            return nothing
        end
    end
    insert!(edge_indices, nindices+1, edge_index)
    return nothing
end

# Intersect a line with materialized edges
# Type-stable
function intersect_edges_explicit(l::LineSegment_2D{F},
                                  edges::Vector{LineSegment_2D{F}}) where {F <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for edge in edges
        npoints, point = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    return sort_intersection_points(l, intersection_points)
end

# Intersect a line with implicitly defined edges
# Type-stable
function intersect_edges_explicit(l::LineSegment_2D{F},
                                  edges::Vector{QuadraticSegment_2D{F}}) where {F <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for edge in edges
        npoints, points = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    return sort_intersection_points(l, intersection_points)
end

# Intersect a line with an implicitly defined edge
# Type-stable
function intersect_edge_implicit(l::LineSegment_2D{F},
                                 edge::SVector{L, U} where {L},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_edge(edge, points)
end

# Intersect a line with a vector of implicitly defined linear edges
# Type-stable
function intersect_edges_implicit(l::LineSegment_2D{F},
                                  edges::Vector{SVector{2, U}},
                                  points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, point = intersect_edge_implicit(l, edge, points)
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    return sort_intersection_points(l, intersection_points) 
end

# Intersect a line with a vector of implicitly defined quadratic edges
# Type-stable
function intersect_edges_implicit(l::LineSegment_2D{F},
                                  edges::Vector{SVector{3, U}},
                                  points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, ipoints = intersect_edge_implicit(l, edge, points)
        if 0 < npoints
            append!(intersection_points, ipoints)
        end
    end
    return sort_intersection_points(l, intersection_points) 
end

# Intersect a line with an implicitly defined face 
# Type-stable
function intersect_face_implicit(l::LineSegment_2D{F},
                                 face::SVector{L, U} where {L},
                                 points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    return l ∩ materialize_face(face, points)
end

# Intersect a line with explicitly defined linear faces
# Type-stable if all faces are one of: linear or quadratic
function intersect_faces_explicit(l::LineSegment_2D{F},
                                  faces::Vector{<:Face_2D{F}} ) where {F <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, points[1:npoints])
        end
    end
    return sort_intersection_points(l, intersection_points)
end

# Intersect a line with implicitly defined faces
# Type-stable if all faces are one of: linear or quadratic
function intersect_faces_implicit(l::LineSegment_2D{F},
                                  faces::Vector{<:SVector{L, U} where L},
                                  points::Vector{Point_2D{F}}
                        ) where {F <: AbstractFloat, U <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{F}[]
    # Intersect the line with each of the faces
    for face in faces
        npoints, ipoints = intersect_face_implicit(l, face, points)
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, ipoints[1:npoints])
        end
    end
    return sort_intersection_points(l, intersection_points)
end

# If a point is a vertex
# Type-stable
function is_vertex(p::Point_2D{F}, points::Vector{Point_2D{F}}) where {F <: AbstractFloat}
    for point in points
        if p ≈ point
            return true
        end
    end
    return false
end

# Return a LineSegment_2D from the point IDs in an edge
# Type-stable
function materialize_edge(edge::SVector{2, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return LineSegment_2D(edge_points(edge, points))
end

# Return a QuadraticSegment_2D from the point IDs in an edge
# Type-stable
function materialize_edge(edge::SVector{3, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return QuadraticSegment_2D(edge_points(edge, points))
end

# Return a materialized edge for each edge in the mesh
# Type-stable if all the edges are the same type
function materialize_edges(edges::Vector{<:SVector{L, U} where {L}},
                           points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return materialize_edge.(edges, Ref(points))
end

# Return a Triangle_2D from the point IDs in a face
# Type-stable
function materialize_face(face::SVector{4, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return Triangle_2D(face_points(face, points))
end

# Return a Quadrilateral_2D from the point IDs in a face
# Type-stable
function materialize_face(face::SVector{5, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return Quadrilateral_2D(face_points(face, points))
end

# Return a Triangle6_2D from the point IDs in a face
# Type-stable
function materialize_face(face::SVector{7, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return Triangle6_2D(face_points(face, points))
end

# Return a Quadrilateral8_2D from the point IDs in a face
# Type-stable
function materialize_face(face::SVector{9, U},
                          points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return Quadrilateral8_2D(face_points(face, points))
end

# Return a materialized face for each face in the mesh
# Type-stable on the condition that the faces are all the same type
function materialize_faces(faces::Vector{<:SVector{L, U} where {L}},
                           points::Vector{Point_2D{F}}) where {F <: AbstractFloat, U <: Unsigned}
    return materialize_face.(faces, Ref(points))
end

# Return the number of edges in a face type
# Type-stable other than the error message
function num_edges(face::SVector{L, U}) where {L, U <: Unsigned}
    cell_type = face[1]
    if cell_type == 5 || cell_type == 22
        return U(3)
    elseif cell_type == 9 || cell_type == 23
        return U(4)
    else
        @error "Unsupported face type"
        return U(0)
    end
end

# Return the ID of the edge shared by two adjacent faces
# Type-stable if all faces are the same type
function shared_edge(face1::U, face2::U,
                     face_edge_connectivity::Vector{<:SVector{N, U} where N},
                    ) where {F <: AbstractFloat, U <: Unsigned}
    edges1 = face_edge_connectivity[face1]
    edges2 = face_edge_connectivity[face1]
    for edge1 in edges1
        for edge2 in edges2
            if edge1 == edge2
                return edge1
            end
        end
    end
    return U(0)
end

# Sort points based on their distance from the start of the line
# Type-stable
function sort_intersection_points(l::LineSegment_2D{F},
                                  points::Vector{Point_2D{F}}) where {F <: AbstractFloat}
    if 0 < length(points)
        # Sort the points based upon their distance to the first point in the line
        distances = distance.(Ref(l.points[1]), points)
        sorted_pairs = sort(collect(zip(distances, points)); by=first)
        # Remove duplicate points
        points_reduced = [sorted_pairs[1][2]]
        npoints = length(sorted_pairs)
        for i = 2:npoints
            if minimum_segment_length < distance(last(points_reduced), sorted_pairs[i][2])
                push!(points_reduced, sorted_pairs[i][2])
            end
        end
        return points_reduced
    else
        return points
    end
end

# Return a mesh with name name, composed of the faces in the set face_ids
# Not type-stable
function submesh(name::String,
                 points::Vector{Point_2D{F}},
                 faces::Vector{<:SVector{L, U} where {L}},
                 face_sets::Dict{String, Set{U}},
                 face_ids::Set{U}) where {F <: AbstractFloat, U <: Unsigned}
    # Setup faces and get all vertex ids
    submesh_faces = Vector{Vector{U}}(undef, length(face_ids))
    vertex_ids = Set{U}()
    for (i, face_id) in enumerate(face_ids)
        face_vec = collect(faces[face_id].data)
        submesh_faces[i] = face_vec
        union!(vertex_ids, Set(face_vec[2:length(face_vec)]))
    end
    # Need to remap vertex ids in faces to new ids
    vertex_ids_sorted = sort(collect(vertex_ids))
    vertex_map = Dict{U, U}()
    for (i,v) in enumerate(vertex_ids_sorted)
        vertex_map[v] = i
    end
    submesh_points = Vector{Point_2D{F}}(undef, length(vertex_ids_sorted))
    for (i, v) in enumerate(vertex_ids_sorted)
        submesh_points[i] = points[v]
    end
    # remap vertex ids in faces
    for face in submesh_faces
        for (i, v) in enumerate(face[2:length(face)])
            face[i + 1] = vertex_map[v]
        end
    end
    # At this point we have points, faces, & name.
    # Just need to get the face sets
    submesh_face_sets = Dict{String, Set{U}}()
    for face_set_name in keys(face_sets)
        set_intersection = face_sets[face_set_name] ∩ face_ids
        if length(set_intersection) !== 0
            submesh_face_sets[face_set_name] = set_intersection
        end
    end
    # Need to remap face ids in face sets
    face_map = Dict{U, U}()
    for (i,f) in enumerate(face_ids)
        face_map[f] = i
    end
    for face_set_name in keys(submesh_face_sets)
        new_set = Set{U}()
        for fid in submesh_face_sets[face_set_name]
            union!(new_set, face_map[fid])
        end
        submesh_face_sets[face_set_name] = new_set
    end
    return UnstructuredMesh_2D{F, U}(name = name,
                                     points = submesh_points,
                                     faces = [SVector{length(f), U}(f) for f in submesh_faces],
                                     face_sets = submesh_face_sets
                                    )
end
