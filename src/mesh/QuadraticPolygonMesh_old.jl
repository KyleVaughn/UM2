struct QuadraticPolygonMesh{T, U} <:QuadraticUnstructuredMesh{2, T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
    face_sets::Dict{String, BitSet}
end

function QuadraticPolygonMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}} = SVector{6, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadraticPolygonMesh(name, points, faces, face_sets)
end

struct QuadraticTriangleMesh{T, U} <:QuadraticUnstructuredMesh{2, T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{SVector{6, U}}
    face_sets::Dict{String, BitSet}
end

function QuadraticTriangleMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{SVector{6, U}} = SVector{6, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadraticTriangleMesh(name, points, faces, face_sets)
end
 
struct QuadraticQuadrilateralMesh{T, U} <:QuadraticUnstructuredMesh{2, T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{SVector{8, U}}
    face_sets::Dict{String, BitSet}
end

function QuadraticQuadrilateralMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{SVector{8, U}} = SVector{8, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadraticQuadrilateralMesh(name, points, faces, face_sets)
end

# Return the edges of the mesh
function edges(mesh::QuadraticUnstructuredMesh{2, T, U}) where {T, U}
    edge_vecs = quadratic_edges.(mesh.faces)
    num_edges = mapreduce(x->length(x), +, edge_vecs)
    edges_unfiltered = Vector{SVector{3, U}}(undef, num_edges)
    iedge = 1
    for edge in edge_vecs
        for i in eachindex(edge)
            if edge[i][1] < edge[i][2]
                edges_unfiltered[iedge] = edge[i]
            else
                edges_unfiltered[iedge] = SVector(edge[i][2], edge[i][1], edge[i][3])
            end
            iedge += 1
        end
    end
    return sort!(unique!(edges_unfiltered))
end

# Return a materialized quadratic triangle
function materialize_face(face_id, mesh::QuadraticTriangleMesh)
    return materialize_quadratic_polygon(mesh.faces[face_id], mesh.points)
end

# Return a materialized quadratic quadrilateral 
function materialize_face(face_id, mesh::QuadraticQuadrilateralMesh)
    return materialize_quadratic_polygon(mesh.faces[face_id], mesh.points)
end

# Return a materialized quadratic polygon 
function materialize_face(face_id, mesh::QuadraticPolygonMesh)
    return materialize_quadratic_polygon(mesh.faces[face_id], mesh.points)
end

# Return a materialized face from the point IDs in a face
function materialize_quadratic_polygon(face::SVector{N}, points::Vector{<:Point}) where {N}
    return QuadraticPolygon{N}(facepoints(face, points))
end

@generated function quadratic_edges(face::SVector{N, U}) where {N, U <:Unsigned}
    M = N ÷ 2
    edges_string = "SVector{$M, SVector{3, U}}("
    for i ∈ 1:M
        id₁ = (i - 1) % M + 1
        id₂ = i % M + 1
        id₃ = i + M
        edges_string *= "SVector{3, U}(face[$id₁], face[$id₂], face[$id₃]), "
    end
    edges_string *= ")"
    return Meta.parse(edges_string)
end

# # A vector of SVectors, denoting the edge ID each face is connected to.
# function face_edge_connectivity(mesh::Quadrilateral8Mesh_2D)
#     if length(mesh.edges) === 0
#         @error "Mesh does not have edges!"
#     end
#     # A vector of MVectors of zeros for each face
#     # Each MVector is the length of the number of edges
#     face_edge = [MVector{4, UInt32}(0, 0, 0, 0) for _ in eachindex(mesh.faces)]
#     # For each face in the mesh, generate the edges.
#     # Search for the index of the edge in the mesh.edges vector
#     # Insert the index of the edge into the face_edge connectivity vector
#     for i in eachindex(mesh.faces)
#         for (j, edge) in enumerate(edges(mesh.faces[i]))
#             face_edge[i][j] = searchsortedfirst(mesh.edges, SVector(edge.data))
#         end
#         if any(x->x === 0x00000000, face_edge[i])
#             @error "Could not determine the face/edge connectivity of face $i"
#         end
#     end
#     return [SVector(sort(conn).data) for conn in face_edge]
# end
# 
# # A vector of SVectors, denoting the edge ID each face is connected to.
# function face_edge_connectivity(mesh::Triangle6Mesh_2D)
#     if length(mesh.edges) === 0
#         @error "Mesh does not have edges!"
#     end
#     # A vector of MVectors of zeros for each face
#     # Each MVector is the length of the number of edges
#     face_edge = [MVector{3, UInt32}(0, 0, 0) for _ in eachindex(mesh.faces)]
#     # For each face in the mesh, generate the edges.
#     # Search for the index of the edge in the mesh.edges vector
#     # Insert the index of the edge into the face_edge connectivity vector
#     for i in eachindex(mesh.faces)
#         for (j, edge) in enumerate(edges(mesh.faces[i]))
#             face_edge[i][j] = searchsortedfirst(mesh.edges, SVector(edge.data))
#         end
#         if any(x->x === 0x00000000, face_edge[i])
#             @error "Could not determine the face/edge connectivity of face $i"
#         end
#     end
#     return [SVector(sort(conn).data) for conn in face_edge]
# end

# How to display a mesh in REPL
function Base.show(io::IO, mesh::QuadraticPolygonMesh)
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
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{6},  mesh.faces))")
    println(io, "  │  └─ Quadrilateral  : $(count(x->x isa SVector{8},  mesh.faces))")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end