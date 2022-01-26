abstract type UnstructuredMesh{Dim,Ord,T,U} end
const UnstructuredMesh2D = UnstructuredMesh{2}
const LinearUnstructuredMesh = UnstructuredMesh{Dim,1} where {Dim}
const LinearUnstructuredMesh2D = UnstructuredMesh{2,1}
const QuadraticUnstructuredMesh = UnstructuredMesh{Dim,2} where {Dim}
const QuadraticUnstructuredMesh2D = UnstructuredMesh{2,2}
Base.broadcastable(mesh::UnstructuredMesh) = Ref(mesh)

# Return a mesh with its edges
function add_edges(mesh::M) where {M <:UnstructuredMesh2D}
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

# Return a mesh with materialized edges
function add_materialized_edges(mesh::M) where {M <: UnstructuredMesh2D}
    if 0 === length(mesh.edges)
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
function add_materialized_faces(mesh::M) where {M <: UnstructuredMesh2D}
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


# SVector of MVectors of point IDs representing the 3 edges of a triangle
function edges(face::SVector{3,U}) where {U <:Unsigned}
    edges = SVector(SVector{2,U}(min(face[1], face[2]), max(face[1], face[2])),  
                    SVector{2,U}(min(face[2], face[3]), max(face[2], face[3])),  
                    SVector{2,U}(min(face[3], face[1]), max(face[3], face[1]))) 
    return edges
end

# SVector of MVectors of point IDs representing the 4 edges of a quadrilateral
function edges(face::SVector{4,U}) where {U <:Unsigned}
    edges = SVector(SVector{2,U}(min(face[1], face[2]), max(face[1], face[2])),  
                    SVector{2,U}(min(face[2], face[3]), max(face[2], face[3])),  
                    SVector{2,U}(min(face[3], face[4]), max(face[3], face[4])),  
                    SVector{2,U}(min(face[4], face[1]), max(face[4], face[1])))  
    return edges
end

# SVector of MVectors of point IDs representing the 3 edges of a quadratic triangle
function edges(face::SVector{6,U}) where {U <:Unsigned}
    edges = SVector(SVector{3,U}(min(face[1], face[2]), max(face[1], face[2]), face[4]),  
                    SVector{3,U}(min(face[2], face[3]), max(face[2], face[3]), face[5]),  
                    SVector{3,U}(min(face[3], face[1]), max(face[3], face[1]), face[6])) 
    return edges
end

# SVector of MVectors of point IDs representing the 4 edges of a quadratic quadrilateral
function edges(face::SVector{8,U}) where {U <:Unsigned}
    edges = SVector(SVector{3,U}(min(face[1], face[2]), max(face[1], face[2]), face[5]),  
                    SVector{3,U}(min(face[2], face[3]), max(face[2], face[3]), face[6]),  
                    SVector{3,U}(min(face[3], face[4]), max(face[3], face[4]), face[7]),  
                    SVector{3,U}(min(face[4], face[1]), max(face[4], face[1]), face[8]))  
    return edges
end

# The unique edges of a mesh 
function edges(mesh::UnstructuredMesh2D)
    edge_vecs = edges.(mesh.faces)
    num_edges = mapreduce(x->length(x),+,edge_vecs)
    edges_unfiltered = Vector{typeof(edge_vecs[1][1])}(undef, num_edges)
    return sort!(unique!(edges_unfiltered))
end

# Return an SVector of the points in the edge (Linear)
function edgepoints(edge::SVector{2}, points::Vector{<:Point})
    return SVector(points[edge[1]], points[edge[2]])
end

# Return an SVector of the points in the edge (Quadratic)
function edgepoints(edge::SVector{3}, points::Vector{<:Point})
    return SVector(points[edge[1]], points[edge[2]], points[edge[3]])
end

# Return an SVector of the points in the edge
function edgepoints(edge_id, mesh::UnstructuredMesh)
    return edgepoints(mesh.edges[edge_id], mesh.points)
end

# Return an SVector of the points in the face (Triangle)
function facepoints(face::SVector{3}, points::Vector{<:Point})
    return SVector(points[face[1]], points[face[2]], points[face[3]])
end

# Return an SVector of the points in the face (Quadrilateral)
function facepoints(face::SVector{4}, points::Vector{<:Point})
    return SVector(points[face[1]], points[face[2]], points[face[3]], points[face[4]])
end

# Return an SVector of the points in the face (Triangle6)
function facepoints(face::SVector{6}, points::Vector{<:Point})
    return SVector(points[face[1]], points[face[2]], points[face[3]],
                   points[face[4]], points[face[5]], points[face[6]])
end

# Return an SVector of the points in the face (Quadrilateral8)
function facepoints(face::SVector{8}, points::Vector{<:Point})
    return SVector(points[face[1]], points[face[2]], points[face[3]], points[face[4]],
                   points[face[5]], points[face[6]], points[face[7]], points[face[8]])
end

# Return an SVector of the points in the face
function facepoints(edge_id, mesh::UnstructuredMesh)
    return facepoints(mesh.faces[edge_id], mesh.points)
end

# Return a LineSegment from the point IDs in an edge
function materialize_edge(edge::SVector{2}, points::Vector{<:Point})
    return LineSegment(edgepoints(edge, points))
end

# Return a QuadraticSegment from the point IDs in an edge
function materialize_edge(edge::SVector{3}, points::Vector{<:Point})
    return QuadraticSegment(edgepoints(edge, points))
end

# Return a LineSegment or QuadraticSegment
function materialize_edge(edge_id, mesh::UnstructuredMesh)
    return materialize_edge(mesh.edges[edge_id], mesh.points)
end

# Return a materialized edge for each edge in the mesh
function materialize_edges(mesh::UnstructuredMesh)
    return materialize_edge.(mesh.edges, Ref(mesh.points))
end

# Return a materialized face from the point IDs in a face
function materialize_face(face::SVector{N}, points::Vector{<:Point}) where {N}
    return Polygon{N}(facepoints(face, points))
end

# Return an SVector of the points in the edge
function materialize_face(face_id, mesh::UnstructuredMesh)
    return materialize_face(mesh.faces[face_id], mesh.points)
end

# Return a materialized face for each face in the mesh
function materialize_faces(mesh::UnstructuredMesh)
    return materialize_face.(mesh.faces, Ref(mesh.points))
end

# Return the number of edges in a face
function num_edges(face::SVector{L,U}) where {L,U}
    if L === 3 || L === 6
        return U(3)
    elseif L === 4 || L === 8
        return U(4)
    else
        return U(0)
    end
end

# How to display a mesh in REPL
function Base.show(io::IO, mesh::UnstructuredMesh)
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
    println(io, "  │  ├─ Linear         : $(count(x->x isa SVector{2},  mesh.edges))")
    println(io, "  │  ├─ Quadratic      : $(count(x->x isa SVector{3},  mesh.edges))")
    println(io, "  │  └─ Materialized?  : $(length(mesh.materialized_edges) !== 0)")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{3},  mesh.faces))")
    println(io, "  │  ├─ Quadrilateral  : $(count(x->x isa SVector{4},  mesh.faces))")
    println(io, "  │  ├─ Triangle6      : $(count(x->x isa SVector{6},  mesh.faces))")
    println(io, "  │  ├─ Quadrilateral8 : $(count(x->x isa SVector{8},  mesh.faces))")
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
