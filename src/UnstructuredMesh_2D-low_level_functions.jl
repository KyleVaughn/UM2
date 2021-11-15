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
