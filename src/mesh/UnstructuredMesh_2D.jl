struct GeneralUnstructuredMesh_2D <: UnstructuredMesh_2D
    name::String
    points::Vector{Point_2D}
    edges::Vector{<:SVector{L, UInt32} where {L}}
    materialized_edges::Vector{<:Edge_2D}
    faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}}
    materialized_faces::Vector{<:Face_2D}
    edge_face_connectivity::Vector{SVector{2, UInt32}}
    face_edge_connectivity::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}}
    boundary_edges::Vector{Vector{UInt32}}
    face_sets::Dict{String, Set{UInt32}}
end

function GeneralUnstructuredMesh_2D(;
        name::String = "DefaultMeshName",
        points::Vector{Point_2D} = Point_2D[],
        edges::Vector{<:SVector{L, UInt32} where {L}} = SVector{2, UInt32}[],
        materialized_edges::Vector{<:Edge_2D} = LineSegment_2D[],
        faces::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}} = SVector{4, UInt32}[],
        materialized_faces::Vector{<:Face_2D} = Triangle_2D[],
        edge_face_connectivity::Vector{SVector{2, UInt32}} = SVector{2, UInt32}[],
        face_edge_connectivity ::Vector{<:SArray{S, UInt32, 1, L} where {S<:Tuple, L}} = SVector{3, UInt32}[],
        boundary_edges::Vector{Vector{UInt32}} = Vector{UInt32}[],
        face_sets::Dict{String, Set{UInt32}} = Dict{String, Set{UInt32}}()
    ) where {F <: AbstractFloat, U <: Unsigned}
        return GeneralUnstructuredMesh_2D(name,
                                          points,
                                          edges,
                                          materialized_edges,
                                          faces,
                                          materialized_faces,
                                          edge_face_connectivity,
                                          face_edge_connectivity,
                                          boundary_edges,
                                          face_sets,
                                         )
end

Base.broadcastable(mesh::GeneralUnstructuredMesh_2D) = Ref(mesh)

# Return a mesh with its edges
function add_edges(mesh::M) where {M <: UnstructuredMesh_2D}
    return M(name = mesh.name,
             points = mesh.points,
             edges = edges(mesh),
             materialized_edges = mesh.materialized_edges,
             faces = mesh.faces,
             materialized_faces = mesh.materialized_faces,
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = mesh.face_edge_connectivity,
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Return a mesh with face/edge connectivity
# and all necessary prerequisites to find the boundary edges
function add_face_edge_connectivity(mesh::M) where {M <: UnstructuredMesh_2D}
    if 0 == length(mesh.edges)
        mesh = add_edges(mesh)
    end
    return M(name = mesh.name,
             points = mesh.points,
             edges = mesh.edges,
             materialized_edges = mesh.materialized_edges,
             faces = mesh.faces,
             materialized_faces = mesh.materialized_faces,
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = face_edge_connectivity(mesh),
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Return a mesh with materialized edges
function add_materialized_edges(mesh::M) where {M <: UnstructuredMesh_2D}
    if 0 == length(mesh.edges)
        mesh = add_edges(mesh)
    end
    return M(name = mesh.name,
             points = mesh.points,
             edges = mesh.edges,
             materialized_edges = materialize_edges(mesh),
             faces = mesh.faces,
             materialized_faces = mesh.materialized_faces,
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = mesh.face_edge_connectivity,
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Return a mesh with materialized faces
function add_materialized_faces(mesh::M) where {M <: UnstructuredMesh_2D}
    return M(name = mesh.name,
             points = mesh.points,
             edges = mesh.edges,
             materialized_edges = mesh.materialized_edges,
             faces = mesh.faces,
             materialized_faces = materialize_faces(mesh),
             edge_face_connectivity = mesh.edge_face_connectivity,
             face_edge_connectivity = mesh.face_edge_connectivity,
             boundary_edges = mesh.boundary_edges,
             face_sets = mesh.face_sets
            )
end

# Area of face
function area(face::SVector{N, UInt32}, points::Vector{Point_2D}) where {N}
    return area(materialize_face(face, points))
end

# Return the area of a face set
function area(mesh::M, face_set::Set{UInt32}) where {M <: UnstructuredMesh_2D} 
    if 0 < length(mesh.materialized_faces)
        return mapreduce(x->area(mesh.materialized_faces[x]), +, face_set)
    else
        return mapreduce(x->area(mesh.faces[x], mesh.points), +, face_set)
    end 
end

# Return the area of a face set by name
function area(mesh::M, set_name::String) where {M <: UnstructuredMesh_2D} 
    return area(mesh, mesh.face_sets[set_name])
end

# Bounding box of a vector of points
function boundingbox(points::Vector{Point_2D})
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return Rectangle_2D(minimum(x), maximum(x), minimum(y), maximum(y))
end

# Bounding box of a vector of points
function boundingbox(points::SVector{L, Point_2D}) where {L}
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return Rectangle_2D(minimum(x), maximum(x), minimum(y), maximum(y))
end

# SVector of MVectors of point IDs representing the 3 edges of a triangle
function edges(face::SVector{3, UInt32})
    edges = SVector( MVector{2, UInt32}(face[1], face[2]),
                     MVector{2, UInt32}(face[2], face[3]),
                     MVector{2, UInt32}(face[3], face[1]) )
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
function edges(face::SVector{4, UInt32})
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
function edges(face::SVector{6, UInt32})
    edges = SVector( MVector{3, UInt32}(face[1], face[2], face[4]),
                     MVector{3, UInt32}(face[2], face[3], face[5]),
                     MVector{3, UInt32}(face[3], face[1], face[6]) )
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
function edges(face::SVector{8, UInt32})
    edges = SVector( MVector{3, UInt32}(face[1], face[2], face[5]),
                     MVector{3, UInt32}(face[2], face[3], face[6]),
                     MVector{3, UInt32}(face[3], face[4], face[7]),
                     MVector{3, UInt32}(face[4], face[1], face[8]) )
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
function edges(mesh::M) where {M <: UnstructuredMesh_2D}
    edges_filtered = sort(unique(reduce(vcat, edges.(mesh.faces))))
    return [ SVector(e.data) for e in edges_filtered ]
end

# Return an SVector of the points in the edge (Linear)
function edge_points(edge::SVector{2, UInt32}, points::Vector{Point_2D})
    return SVector(points[edge[1]], points[edge[2]])
end

# Return an SVector of the points in the edge (Quadratic)
function edge_points(edge::SVector{3, UInt32}, points::Vector{Point_2D})
    return SVector(points[edge[1]], points[edge[2]], points[edge[3]])
end

# Return an SVector of the points in the edge
function edge_points(edge_id::UInt32, mesh::M) where {M <: UnstructuredMesh_2D}
    return edge_points(mesh.edges[edge_id], mesh.points)
end

# Return an SVector of the points in the face (Triangle)
function face_points(face::SVector{3, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[1]], points[face[2]], points[face[3]])
end

# Return an SVector of the points in the face (Quadrilateral)
function face_points(face::SVector{4, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[1]], points[face[2]], points[face[3]], points[face[4]])
end

# Return an SVector of the points in the face (Triangle6)
function face_points(face::SVector{6, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[1]], points[face[2]], points[face[3]],
                   points[face[4]], points[face[5]], points[face[6]])
end

# Return an SVector of the points in the face (Quadrilateral8)
function face_points(face::SVector{8, UInt32}, points::Vector{Point_2D})
    return SVector(points[face[1]], points[face[2]], points[face[3]], points[face[4]],
                   points[face[5]], points[face[6]], points[face[7]], points[face[8]])
end

# Return an SVector of the points in the face
function face_points(edge_id::UInt32, mesh::M) where {M <: UnstructuredMesh_2D}
    return face_points(mesh.faces[edge_id], mesh.points)
end

# Return a LineSegment_2D from the point IDs in an edge
function materialize_edge(edge::SVector{2, UInt32}, points::Vector{Point_2D})
    return LineSegment_2D(edge_points(edge, points))
end

# Return a QuadraticSegment_2D from the point IDs in an edge
function materialize_edge(edge::SVector{3, UInt32}, points::Vector{Point_2D})
    return QuadraticSegment_2D(edge_points(edge, points))
end

# Return a LineSegment_2D or QuadraticSegment_2D
function materialize_edge(edge_id::UInt32, mesh::M) where {M <: UnstructuredMesh_2D}
    return materialize_edge(mesh.edges[edge_id], mesh.points)
end

# Return a materialized edge for each edge in the mesh
function materialize_edges(mesh::M) where {M <: UnstructuredMesh_2D}
    return materialize_edge.(mesh.edges, Ref(mesh.points))
end

# Return a Triangle_2D from the point IDs in a face
function materialize_face(face::SVector{3, UInt32}, points::Vector{Point_2D})
    return Triangle_2D(face_points(face, points))
end

# Return a Quadrilateral_2D from the point IDs in a face
function materialize_face(face::SVector{4, UInt32}, points::Vector{Point_2D})
    return Quadrilateral_2D(face_points(face, points))
end

# Return a Triangle6_2D from the point IDs in a face
function materialize_face(face::SVector{6, UInt32}, points::Vector{Point_2D})
    return Triangle6_2D(face_points(face, points))
end

# Return a Quadrilateral8_2D from the point IDs in a face
function materialize_face(face::SVector{8, UInt32}, points::Vector{Point_2D})
    return Quadrilateral8_2D(face_points(face, points))
end

# Return an SVector of the points in the edge
function materialize_face(face_id::UInt32, mesh::M) where {M <: UnstructuredMesh_2D}
    return materialize_face(mesh.faces[face_id], mesh.points)
end

# Return a materialized face for each face in the mesh
function materialize_faces(mesh::M) where {M <: UnstructuredMesh_2D}
    return materialize_face.(mesh.faces, Ref(mesh.points))
end

# Return the number of edges in a face
function num_edges(face::SVector{L, UInt32}) where {L}
    if L % 3 === 0 
        return 0x00000003
    elseif L % 4 === 0
        return 0x00000004
    else
        # Error
        return 0x00000000
    end
end

# How to display a mesh in REPL
function Base.show(io::IO, mesh::M) where {M <: UnstructuredMesh_2D}
    mesh_type = typeof(mesh)
    println(io, mesh_type)
    println(io, "  ├─ Name      : $(mesh.name)")
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        size_KB = size_MB*1000
        println(io, "  ├─ Size (KB) : $size_KB")
    else
        println(io, "  ├─ Size (MB) : $size_MB")
    end
    println(io, "  ├─ Points    : $(length(mesh.points))")
    nedges = length(mesh.edges)
    println(io, "  ├─ Edges     : $nedges")
    println(io, "  │  ├─ Linear         : $(count(x->x isa SVector{2, UInt32},  mesh.edges))")
    println(io, "  │  ├─ Quadratic      : $(count(x->x isa SVector{3, UInt32},  mesh.edges))")
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_edges) !== 0)")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{3, UInt32},  mesh.faces))")
    println(io, "  │  ├─ Quadrilateral  : $(count(x->x isa SVector{4, UInt32},  mesh.faces))")
    println(io, "  │  ├─ Triangle6      : $(count(x->x isa SVector{6, UInt32},  mesh.faces))")
    println(io, "  │  ├─ Quadrilateral8 : $(count(x->x isa SVector{8, UInt32},  mesh.faces))")
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_faces) !== 0)")
    println(io, "  ├─ Connectivity")
    println(io, "  │  ├─ Edge/Face : $(0 < length(mesh.edge_face_connectivity))")
    println(io, "  │  └─ Face/Edge : $(0 < length(mesh.face_edge_connectivity))")
    println(io, "  ├─ Boundary edges")
    nsides = length(mesh.boundary_edges)
    if nsides !== 0
        println(io, "  │  ├─ Edges : $(mapreduce(x->length(x), +, mesh.boundary_edges))")
    else
        println(io, "  │  ├─ Edges : 0") 
    end
    println(io, "  │  └─ Sides : $nsides")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end
