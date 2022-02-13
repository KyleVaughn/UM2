struct QuadraticPolygonMesh{Dim, T, U} <:QuadraticUnstructuredMesh{Dim, T, U}
    name::String
    points::Vector{Point{Dim, T}}
    edges::Vector{SVector{3, U}}
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
    face_sets::Dict{String, BitSet}
end

function QuadraticPolygonMesh{Dim, T, U}(;
    name::String = "default_name",
    points::Vector{Point{Dim, T}} = Point{Dim, T}[],
    edges::Vector{SVector{3, U}} = SVector{3, U}[],
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}} = SVector{6, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {Dim, T, U}
    return QuadraticPolygonMesh(name, points, edges, faces, face_sets)
end

struct QuadraticTriangleMesh{Dim, T, U} <:QuadraticUnstructuredMesh{Dim, T, U}
    name::String
    points::Vector{Point{Dim, T}}
    edges::Vector{SVector{3, U}}
    faces::Vector{SVector{6, U}}
    face_sets::Dict{String, BitSet}
end

function QuadraticTriangleMesh{Dim, T, U}(;
    name::String = "default_name",
    points::Vector{Point{Dim, T}} = Point{Dim, T}[],
    edges::Vector{SVector{3, U}} = SVector{3, U}[],
    faces::Vector{SVector{6, U}} = SVector{6, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {Dim, T, U}
    return QuadraticTriangleMesh(name, points, edges, faces, face_sets)
end
 
struct QuadraticQuadrilateralMesh{Dim, T, U} <:QuadraticUnstructuredMesh{Dim, T, U}
    name::String
    points::Vector{Point{Dim, T}}
    edges::Vector{SVector{3, U}}
    faces::Vector{SVector{8, U}}
    face_sets::Dict{String, BitSet}
end

function QuadraticQuadrilateralMesh{Dim, T, U}(;
    name::String = "default_name",
    points::Vector{Point{Dim, T}} = Point{Dim, T}[],
    edges::Vector{SVector{3, U}} = SVector{3, U}[],
    faces::Vector{SVector{8, U}} = SVector{8, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {Dim, T, U}
    return QuadraticQuadrilateralMesh(name, points, edges, faces, face_sets)
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

# # Axis-aligned bounding box, in 2d a rectangle.
# function boundingbox(mesh::QuadraticUnstructuredMesh_2D; boundary_shape::String="Unknown")
#     if boundary_shape == "Rectangle"
#         return boundingbox(mesh.points)
#     else
#         # Currently only polygons, so can use the points
#         nsides = length(mesh.boundary_edges)
#         if nsides !== 0
#             boundary_edge_IDs = reduce(vcat, mesh.boundary_edges)
#             point_IDs = reduce(vcat, mesh.edges[boundary_edge_IDs])
#             return boundingbox(mesh.points[point_IDs]) 
#         else
#             return reduce(union, boundingbox.(materialize_edge.(edges(mesh), Ref(mesh.points))))
#         end
#     end
# end
#  
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
    nedges = length(mesh.edges)
    println(io, "  ├─ Edges     : $nedges")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{6},  mesh.faces))")
    println(io, "  │  └─ Quadrilateral  : $(count(x->x isa SVector{8},  mesh.faces))")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end
