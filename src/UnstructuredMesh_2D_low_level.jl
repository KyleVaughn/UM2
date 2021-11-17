# Area of a triangle
function area(mesh::UnstructuredMesh_2D{T, I}, face::NTuple{4, I}) where {T <: AbstractFloat, I <: Unsigned}
    return area(materialize_face(mesh, face))
end

# Area of a quadrilateral
function area(mesh::UnstructuredMesh_2D{T, I}, face::NTuple{5, I}) where {T <: AbstractFloat, I <: Unsigned}
    return area(materialize_face(mesh, face))
end

# Area of a quadratic triangle
function area(mesh::UnstructuredMesh_2D{T, I}, face::NTuple{7, I}) where {T <: AbstractFloat, I <: Unsigned}
    return area(materialize_face(mesh, face))
end

# Area of a quadratic quadrilateral
function area(mesh::UnstructuredMesh_2D{T, I}, face::NTuple{9, I}) where {T <: AbstractFloat, I <: Unsigned}
    return area(materialize_face(mesh, face))
end

# Return a vector containing vectors of the edges in each side of the mesh's bounding shape, e.g.
# For a rectangular bounding shape the sides are North, East, South, West. Then the output would
# be [ [e1, e2, e3, ...], [e17, e18, e18, ...], ..., [e100, e101, ...]]
function boundary_edges(mesh::UnstructuredMesh_2D{T, I};
                       bounding_shape="Rectangle") where {T<:AbstractFloat, I <: Unsigned}
    # edges which have face 0 in their edge_face connectivity are boundary edges
    boundary_edges = findall(x->x[1] == 0, mesh.edge_face_connectivity)
    if bounding_shape == "Rectangle"
        # Sort edges into NESW
        bb = bounding_box(mesh, rectangular_boundary=true)
        y_north = bb.points[3].x[2]
        x_east  = bb.points[3].x[1]
        y_south = bb.points[1].x[2]
        x_west  = bb.points[1].x[1]
        p_NW = bb.points[4]
        p_NE = bb.points[3]
        p_SE = bb.points[2]
        p_SW = bb.points[1]
        edges_north = I[]
        edges_east = I[]
        edges_south = I[]
        edges_west = I[]
        # Insert edges so that indices move from NW -> NE -> SE -> SW -> NW
        for i = 1:length(boundary_edges)
            edge = I(boundary_edges[i])
            epoints = edge_points(mesh, mesh.edges[edge])
            if all(x->abs(x[2] - y_north) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_north, p_NW, mesh)
            elseif all(x->abs(x[1] - x_east) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_east, p_NE, mesh)
            elseif all(x->abs(x[2] - y_south) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_south, p_SE, mesh)
            elseif all(x->abs(x[1] - x_west) < 1e-4, epoints)
                insert_boundary_edge!(edge, edges_west, p_SW, mesh)
            else
                @error "Edge $iedge could not be classified as NSEW"
            end
        end
        return [ edges_north, edges_east, edges_south, edges_west ]
    else
        return [ convert(Vector{I}, boundary_edges) ]
    end
end

# Vector of vector of point IDs representing the 3 edges of a triangle
function edges(face::NTuple{4, I}) where {I <: Unsigned} 
    cell_type = face[1]
    if cell_type ∈  UnstructuredMesh_2D_linear_cell_types 
        edges = [ [face[2], face[3]],
                  [face[3], face[4]],
                  [face[4], face[2]] ]
    else
        @error "Unsupported cell type"
        edges = [[I(0), I(0)]]
    end
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

# Vector of vector of point IDs representing the 4 edges of a quadrilateral
function edges(face::NTuple{5, I}) where {I <: Unsigned}
    cell_type = face[1]
    if cell_type ∈  UnstructuredMesh_2D_linear_cell_types 
        edges = [ [face[2], face[3]],
                  [face[3], face[4]],
                  [face[4], face[5]],
                  [face[5], face[2]] ]
    else
        @error "Unsupported cell type"
        edges = [[I(0), I(0)]]
    end
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

# Vector of vector of point IDs representing the 4 edges of a quadratic quadrilateral
function edges(face::NTuple{7, I}) where {I <: Unsigned} 
    cell_type = face[1]
    if cell_type ∈  UnstructuredMesh_2D_quadratic_cell_types
        edges = [ [face[2], face[3], face[5]],
                  [face[3], face[4], face[6]],
                  [face[4], face[2], face[7]] ]             
    else
        @error "Unsupported cell type"
        edges = [[I(0), I(0)]]
    end
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

# The unique edges from a vector of triangles or quadrilaterals represented by point IDs
function edges(faces::Vector{<:Union{NTuple{4, I}, NTuple{5, I}}}) where {I <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(collect(Set{Vector{I}}(edges_unfiltered)))
    return [ Tuple(e) for e in edges_filtered ]::Vector{NTuple{2, I}}
end

# The unique edges from a vector of quadratic triangles or quadratic quadrilaterals 
# represented by point IDs
function edges(faces::Vector{<:Union{NTuple{7, I}, NTuple{9, I}}}) where {I <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(collect(Set{Vector{I}}(edges_unfiltered)))
    return [ Tuple(e) for e in edges_filtered ]::Vector{NTuple{3, I}}
end

# The unique edges from a vector of faces represented by point IDs
function edges(faces::Vector{<:Tuple{Vararg{I, N} where N}}) where {I <: Unsigned}
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(collect(Set{Vector{I}}(edges_unfiltered)))
    return [ Tuple(e) for e in edges_filtered ]
end

# Find the faces which share the vertex of ID p.
function faces_sharing_vertex(p::P,
        faces::Vector{<:Tuple{Vararg{I, N} where N}}) where {P <: Integer, I <: Unsigned}
    shared_faces = Int64[]
    for i = 1:length(faces)
        N = length(faces[i])
        if p ∈  faces[i][2:N]
            push!(shared_faces, i)
        end
    end
    return shared_faces
end

# Find the face containing the point p, with explicitly represented faces
function find_face_explicit(p::Point_2D{T},
                            faces::Vector{<:Face_2D{T}}
                           ) where {T <: AbstractFloat}
    for i = 1:length(faces)
        if p ∈  faces[i]
            return i
        end
    end
    @error "Could not find face for point $p"
    return 0
end

# Return the face containing the point p, with implicitly represented faces
function find_face_implicit(p::Point_2D{T},
                            mesh::UnstructuredMesh_2D{T, I},
                            faces::Vector{<:Tuple{Vararg{I, N} where N}}
                            ) where {T <: AbstractFloat, I <: Unsigned}
    for i = 1:length(faces)
        bool = p ∈  materialize_face(mesh, faces[i])
        if bool
            return i
        end
    end
    @error "Could not find face for point $p"
    return 0
end

# Return a vector if the faces adjacent to the face of ID face
function adjacent_faces(face::I,
                        face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}},
                        edge_face_connectivity::Vector{NTuple{2, I}}
                        )where {I <: Unsigned}
    edges = face_edge_connectivity[face]
    the_adjacent_faces = I[]
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

# Return a tuple of the points in the edge
function edge_points(mesh::UnstructuredMesh_2D{T, I},
                     edge::NTuple{2, I}) where {T <: AbstractFloat, I <: Unsigned}
    return (mesh.points[edge[1]], mesh.points[edge[2]])
end

# Return a tuple of the points in the edge
function edge_points(mesh::UnstructuredMesh_2D{T, I},
                     edge::NTuple{3, I}) where {T <: AbstractFloat, I <: Unsigned}
    return (mesh.points[edge[1]],
            mesh.points[edge[2]],
            mesh.points[edge[3]]
           )
end

# Return a tuple of the points in the face 
function face_points(mesh::UnstructuredMesh_2D{T, I},
                     face::NTuple{4, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:4])])::NTuple{3, Point_2D{T}}
end

# Return a tuple of the points in the face 
function face_points(mesh::UnstructuredMesh_2D{T, I},
                     face::NTuple{5, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:5])])::NTuple{4, Point_2D{T}}
end

# Return a tuple of the points in the face 
function face_points(mesh::UnstructuredMesh_2D{T, I},
                     face::NTuple{7, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:7])])::NTuple{6, Point_2D{T}}
end

# Return a tuple of the points in the face 
function face_points(mesh::UnstructuredMesh_2D{T, I},
                     face::NTuple{9, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Tuple(mesh.points[collect(face[2:9])])::NTuple{8, Point_2D{T}}
end

# Insert the boundary edge into the correct place in the vector of edge indices, based on
# the distance from some reference point
function insert_boundary_edge!(edge_index::I, edge_indices::Vector{I}, p_ref::Point_2D{T},
        mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    # Compute the minimum distance from the edge to be inserted to the reference point
    epoints = edge_points(mesh, mesh.edges[edge_index])
    insertion_distance = minimum([ distance(p_ref, p_edge) for p_edge in epoints ])
    # Loop through the edge indices until an edge with greater distance from the reference point
    # is found, then insert
    nindices = length(edge_indices)
    for i = 1:nindices
        edge = edge_indices[i]
        epoints = edge_points(mesh, mesh.edges[edge])
        edge_distance = minimum([ distance(p_ref, p_edge) for p_edge in epoints ])
        if insertion_distance < edge_distance
            insert!(edge_indices, i, edge_index)
            return nothing
        end
    end
    insert!(edge_indices, nindices+1, edge_index)
    return nothing
end

# Intersect a line with the edges of a mesh
function intersect_edges(l::LineSegment_2D{T},
                         mesh::UnstructuredMesh_2D{T, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # Implicit intersection has been faster in every test, so this is the default
    if length(mesh.materialized_edges) !== 0
        intersection_points = intersect_edges_explicit(l, mesh.materialized_edges)
        return sort_intersection_points(l, intersection_points)
    else
        intersection_points = intersect_edges_implicit(l, mesh, mesh.edges)
        return sort_intersection_points(l, intersection_points)
    end
end

# Intersect a line with materialized edges
function intersect_edges_explicit(l::LineSegment_2D{T},
                                  edges::Vector{LineSegment_2D{T}}) where {T <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for edge in edges
        npoints, point = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    return intersection_points
end

# Intersect a line with implicitly defined edges
function intersect_edges_explicit(l::LineSegment_2D{T},
                                  edges::Vector{QuadraticSegment_2D{T}}) where {T <: AbstractFloat}
    # A vector to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for edge in edges
        npoints, points = l ∩ edge
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

# Intersect a line with an implicitly defined linear edge
function intersect_edge_implicit(l::LineSegment_2D{T},
                                 mesh::UnstructuredMesh_2D{T, I},
                                 edge::NTuple{2, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    return l ∩ LineSegment_2D(edge_points(mesh, edge))
end

# Intersect a line with an implicitly defined quadratic edge
function intersect_edge_implicit(l::LineSegment_2D{T},
                                 mesh::UnstructuredMesh_2D{T, I},
                                 edge::NTuple{3, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    return l ∩ QuadraticSegment_2D(edge_points(mesh, edge))
end

# Intersect a line with a vector of implicitly defined linear edges
function intersect_edges_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  edges::Vector{<:Union{NTuple{2, I}, NTuple{3, I}}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, point = intersect_edge_implicit(l, mesh, edge)
        if 0 < npoints
            push!(intersection_points, point)
        end
    end
    return intersection_points
end

# Intersect a line with a vector of implicitly defined quadratic edges
function intersect_edges_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  edges::Vector{NTuple{3, I}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for edge in edges
        npoints, points = l ∩ QuadraticSegment_2D(edge_points(mesh, edge))
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

# Intersect a line with an implicitly defined Triangle_2D
function intersect_face_implicit(l::LineSegment_2D{T},
                                 mesh::UnstructuredMesh_2D{T, I},
                                 face::NTuple{4, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    return l ∩ Triangle_2D(face_points(mesh, face))
end

# Intersect a line with an implicitly defined Quadrilateral_2D
function intersect_face_implicit(l::LineSegment_2D{T},
                                 mesh::UnstructuredMesh_2D{T, I},
                                 face::NTuple{5, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    return l ∩ Quadrilateral_2D(face_points(mesh, face))
end

# Intersect a line with an implicitly defined Triangle6_2D
function intersect_face_implicit(l::LineSegment_2D{T},
                                 mesh::UnstructuredMesh_2D{T, I},
                                 face::NTuple{7, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    return l ∩ Triangle6_2D(face_points(mesh, face))
end

# Intersect a line with an implicitly defined Quadrilateral8_2D
function intersect_face_implicit(l::LineSegment_2D{T},
                                 mesh::UnstructuredMesh_2D{T, I},
                                 face::NTuple{9, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    return l ∩ Quadrilateral8_2D(face_points(mesh, face))
end

# Intersect a line with the faces of a mesh
function intersect_faces(l::LineSegment_2D{T},
                         mesh::UnstructuredMesh_2D{T, I}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # Explicit faces have been faster in every test, so this is the default
    if length(mesh.materialized_faces) !== 0
        intersection_points = intersect_faces_explicit(l, mesh.materialized_faces)
        return sort_intersection_points(l, intersection_points)
    else
        # Check if any of the face types are unsupported
        unsupported = count(x->x[1] ∉  UnstructuredMesh_2D_cell_types, mesh.faces)
        if 0 < unsupported
            @error "Mesh contains an unsupported face type"
        end
        intersection_points = intersect_faces_implicit(l, mesh, mesh.faces)
        return sort_intersection_points(l, intersection_points)
    end
end

# Intersect a line with explicitly defined linear faces
function intersect_faces_explicit(l::LineSegment_2D{T},
                                  faces::Vector{<:Union{Triangle_2D{T}, Quadrilateral_2D{T}}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

# Intersect a line with explicitly defined quadratic faces
function intersect_faces_explicit(l::LineSegment_2D{T},
                                  faces::Vector{<:Union{Triangle6_2D{T}, Quadrilateral8_2D{T}}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

# Intersect a line with explicitly defined faces
function intersect_faces_explicit(l::LineSegment_2D{T},
                                  faces::Vector{<:Face_2D{T}}
                        ) where {T <: AbstractFloat}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    for face in faces
        npoints, points = l ∩ face
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

# Intersect a line with implicitly defined faces
function intersect_faces_implicit(l::LineSegment_2D{T},
                                  mesh::UnstructuredMesh_2D{T, I},
                                  faces::Vector{<:Tuple{Vararg{I, N} where N}}
                        ) where {T <: AbstractFloat, I <: Unsigned}
    # An array to hold all of the intersection points
    intersection_points = Point_2D{T}[]
    # Intersect the line with each of the faces
    for face in faces
        npoints, points = intersect_face_implicit(l, mesh, face)
        # If the intersections yields 1 or more points, push those points to the array of points
        if 0 < npoints
            append!(intersection_points, collect(points[1:npoints]))
        end
    end
    return intersection_points
end

# If a point is a vertex
function is_vertex(p::Point_2D{T}, mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    for point in mesh.points
        if p ≈ point
            return true
        end
    end
    return false
end

# Return a LineSegment_2D from the point IDs in an edge
function materialize_edge(mesh::UnstructuredMesh_2D{T, I},
                          edge::NTuple{2, I}) where {T <: AbstractFloat, I <: Unsigned}
    return LineSegment_2D(edge_points(mesh, edge))
end

# Return a QuadraticSegment_2D from the point IDs in an edge
function materialize_edge(mesh::UnstructuredMesh_2D{T, I}, 
                          edge::NTuple{3, I}) where {T <: AbstractFloat, I <: Unsigned}
    return QuadraticSegment_2D(edge_points(mesh, edge))
end

# Return a materialized edge for each edge in the mesh
function materialize_edges(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    return materialize_edge.(mesh, mesh.edges)
end

# Return a Triangle_2D from the point IDs in a face
function materialize_face(mesh::UnstructuredMesh_2D{T, I}, 
                          face::NTuple{4, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Triangle_2D(face_points(mesh, face))
end

# Return a Quadrilateral_2D from the point IDs in a face
function materialize_face(mesh::UnstructuredMesh_2D{T, I}, 
                          face::NTuple{5, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Quadrilateral_2D(face_points(mesh, face))
end

# Return a Triangle6_2D from the point IDs in a face
function materialize_face(mesh::UnstructuredMesh_2D{T, I}, 
                          face::NTuple{7, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Triangle6_2D(face_points(mesh, face))
end

# Return a Quadrilateral8_2D from the point IDs in a face
function materialize_face(mesh::UnstructuredMesh_2D{T, I}, 
                          face::NTuple{9, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Quadrilateral8_2D(face_points(mesh, face))
end

# Return a materialized face for each face in the mesh
function materialize_faces(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat,
                                                                    I <: Unsigned}
    return materialize_face.(mesh, mesh.faces)::Vector{<:Face_2D{T}}
end

# Return the number of edges in a face type
function num_edges(face::Tuple{Vararg{I}}) where {I <: Unsigned}
    cell_type = face[1]
    if cell_type == 5 || cell_type == 22
        return I(3)
    elseif cell_type == 9 || cell_type == 23
        return I(4)
    else
        @error "Unsupported face type"
        return I(0)
    end
end

# Return the ID of the edge shared by two adjacent faces
function shared_edge(face_edge_connectivity::Vector{<:Tuple{Vararg{I, M} where M}},
                     face1::I, face2::I) where {T <: AbstractFloat, I <: Unsigned}
    edges1 = face_edge_connectivity[face1]
    edges2 = face_edge_connectivity[face1]
    for edge1 in edges1
        for edge2 in edges2
            if edge1 == edge2
                return edge1
            end
        end
    end
    return I(0)
end

# Sort points based on their distance from the start of the line
function sort_intersection_points(l::LineSegment_2D{T},
                                  points::Vector{Point_2D{T}}) where {T <: AbstractFloat}
    if 0 < length(points)
        # Sort the points based upon their distance to the first point in the line
        distances = distance.(l.points[1], points)
        sorted_pairs = sort(collect(zip(distances, points)); by=first)
        # Remove duplicate points
        points_reduced::Vector{Point_2D{T}} = [sorted_pairs[1][2]]
        npoints::Int64 = length(sorted_pairs)
        for i = 2:npoints
            if minimum_segment_length < distance(last(points_reduced), sorted_pairs[i][2])
                push!(points_reduced, sorted_pairs[i][2])
            end
        end
        return points_reduced::Vector{Point_2D{T}}
    else
        return points
    end
end

# Return a mesh with name name, composed of the faces in the set face_ids
function submesh(mesh::UnstructuredMesh_2D{T, I},
                 face_ids::Set{I};
                 name::String = "DefaultMeshName") where {T<:AbstractFloat, I <: Unsigned}
    # Setup faces and get all vertex ids
    faces = Vector{Vector{I}}(undef, length(face_ids))
    vertex_ids = Set{I}()
    for (i, face_id) in enumerate(face_ids)
        face = collect(mesh.faces[face_id])
        faces[i] = face
        union!(vertex_ids, Set{I}(face[2:length(face)]))
    end
    # Need to remap vertex ids in faces to new ids
    vertex_ids_sorted = sort(collect(vertex_ids))
    vertex_map = Dict{I, I}()
    for (i,v) in enumerate(vertex_ids_sorted)
        vertex_map[v] = i
    end
    points = Vector{Point_2D{T}}(undef, length(vertex_ids_sorted))
    for (i, v) in enumerate(vertex_ids_sorted)
        points[i] = mesh.points[v]
    end
    # remap vertex ids in faces
    for face in faces
        for (i, v) in enumerate(face[2:length(face)])
            face[i + 1] = vertex_map[v]
        end
    end
    # At this point we have points, faces, & name.
    # Just need to get the face sets
    face_sets = Dict{String, Set{I}}()
    for face_set_name in keys(mesh.face_sets)
        set_intersection = intersect(mesh.face_sets[face_set_name], face_ids)
        if length(set_intersection) !== 0
            face_sets[face_set_name] = set_intersection
        end
    end
    # Need to remap face ids in face sets
    face_map = Dict{I, I}()
    for (i,f) in enumerate(face_ids)
        face_map[f] = i
    end
    for face_set_name in keys(face_sets)
        new_set = Set{I}()
        for fid in face_sets[face_set_name]
            union!(new_set, face_map[fid])
        end
        face_sets[face_set_name] = new_set
    end
    return UnstructuredMesh_2D{T, I}(name = name,
                                     points = points,
                                     faces = [Tuple(f) for f in faces],
                                     face_sets = face_sets
                                    )
end

