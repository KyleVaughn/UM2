abstract type Mesh_2D end

# SVector of MVectors of point IDs representing the 3 edges of a triangle
# Type-stable
function edges(face::SVector{4, UInt32})
    edges = SVector( MVector{2, UInt32}(face[2], face[3]),
                     MVector{2, UInt32}(face[3], face[4]),
                     MVector{2, UInt32}(face[4], face[2]) )
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
function edges(face::SVector{5, UInt32})
    edges = SVector( MVector{2, UInt32}(face[2], face[3]),
                     MVector{2, UInt32}(face[3], face[4]),
                     MVector{2, UInt32}(face[4], face[5]),
                     MVector{2, UInt32}(face[5], face[2]) )
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
function edges(face::SVector{7, UInt32})
    edges = SVector( MVector{3, UInt32}(face[2], face[3], face[5]),
                     MVector{3, UInt32}(face[3], face[4], face[6]),
                     MVector{3, UInt32}(face[4], face[2], face[7]) )
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
function edges(face::SVector{9, UInt32})
    edges = SVector( MVector{3, UInt32}(face[2], face[3], face[6]),
                     MVector{3, UInt32}(face[3], face[4], face[7]),
                     MVector{3, UInt32}(face[4], face[5], face[8]),
                     MVector{3, UInt32}(face[5], face[2], face[9]) )
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
function edges(faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}})
    edge_arr = edges.(faces)
    edges_unfiltered = [ edge for edge_vec in edge_arr for edge in edge_vec ]
    # Filter the duplicate edges
    edges_filtered = sort(unique(edges_unfiltered))
    return [ SVector(e.data) for e in edges_filtered ]
end


# Return an SVector of the points in the linear edge
# Type-stable
function edge_points(edge::SVector{2, UInt32}, points::Vector{Point_2D})
    return SVector(points[edge[1]], points[edge[2]])
end

# Return an SVector of the points in the quadratic edge
# Type-stable
function edge_points(edge::SVector{3, UInt32}, points::Vector{Point_2D})
    return SVector(points[edge[1]], points[edge[2]], points[edge[3]])
end

# Return an SVector of the points in the triangle face
# Type-stable
function face_points(face::SVector{4, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[2]], points[face[3]], points[face[4]])
end

# Return an SVector of the points in the quadrilateral face
# Type-stable
function face_points(face::SVector{5, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[2]], points[face[3]], points[face[4]], points[face[5]])
end

# Return an SVector of the points in the 2nd order triangle face
# Type-stable
function face_points(face::SVector{7, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[2]], points[face[3]], points[face[4]], 
                   points[face[5]], points[face[6]], points[face[7]])
end

# Return an SVector of the points in the 2nd order quadrilateral face
# Type-stable
function face_points(face::SVector{9, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[2]], points[face[3]], points[face[4]], points[face[5]], 
                   points[face[6]], points[face[7]], points[face[8]], points[face[9]])
end

# Return a LineSegment_2D from the point IDs in an edge
# Type-stable
function materialize_edge(edge::SVector{2, UInt32}, points::Vector{Point_2D})
    return LineSegment_2D(edge_points(edge, points))
end

# Return a QuadraticSegment_2D from the point IDs in an edge
# Type-stable
function materialize_edge(edge::SVector{3, UInt32}, points::Vector{Point_2D})
    return QuadraticSegment_2D(edge_points(edge, points))
end

# Return a materialized edge for each edge in the mesh
# Type-stable if all the edges are the same type
function materialize_edges(edges::Vector{<:SVector{L, UInt32} where {L}},
                           points::Vector{Point_2D})
    return materialize_edge.(edges, Ref(points))
end
