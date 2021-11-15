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
        bb = AABB(mesh, rectangular_boundary=true)
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
            iedge = I(boundary_edges[i])
            edge_points = get_edge_points(mesh, mesh.edges[iedge])
            if all(x->abs(x[2] - y_north) < 1e-4, edge_points)
                insert_boundary_edge!(iedge, edges_north, p_NW, mesh)
            elseif all(x->abs(x[1] - x_east) < 1e-4, edge_points)
                insert_boundary_edge!(iedge, edges_east, p_NE, mesh)
            elseif all(x->abs(x[2] - y_south) < 1e-4, edge_points)
                insert_boundary_edge!(iedge, edges_south, p_SE, mesh)
            elseif all(x->abs(x[1] - x_west) < 1e-4, edge_points)
                insert_boundary_edge!(iedge, edges_west, p_SW, mesh)
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
        bool = p ∈  materialize_face(mesh, face)
        if bool
            return i
        end
    end
    @error "Could not find face for point $p"
    return 0
end

# Return a LineSegment_2D from the point IDs in an edge
function materialize_edge(mesh::UnstructuredMesh_2D{T, I},
                          edge::NTuple{2, I}) where {T <: AbstractFloat, I <: Unsigned}
    return LineSegment_2D(get_edge_points(mesh, edge))
end

# Return a QuadraticSegment_2D from the point IDs in an edge
function materialize_edge(mesh::UnstructuredMesh_2D{T, I}, 
                          edge::NTuple{3, I}) where {T <: AbstractFloat, I <: Unsigned}
    return QuadraticSegment_2D(get_edge_points(mesh, edge))
end

# Return a materialized edge for each edge in the mesh
function materialize_edges(mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    return materialize_edge.(mesh, mesh.edges)
end

# Return a Triangle_2D from the point IDs in a face
function materialize_face(mesh::UnstructuredMesh_2D{T, I}, 
                          face::NTuple{4, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Triangle_2D(get_face_points(mesh, face))
end

# Return a Quadrilateral_2D from the point IDs in a face
function materialize_face(mesh::UnstructuredMesh_2D{T, I}, 
                          face::NTuple{5, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Quadrilateral_2D(get_face_points(mesh, face))
end

# Return a Triangle6_2D from the point IDs in a face
function materialize_face(mesh::UnstructuredMesh_2D{T, I}, 
                          face::NTuple{7, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Triangle6_2D(get_face_points(mesh, face))
end

# Return a Quadrilateral8_2D from the point IDs in a face
function materialize_face(mesh::UnstructuredMesh_2D{T, I}, 
                          face::NTuple{9, I}) where {T <: AbstractFloat, I <: Unsigned}
    return Quadrilateral8_2D(get_face_points(mesh, face))
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
