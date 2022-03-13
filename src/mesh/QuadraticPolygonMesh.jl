struct QuadraticPolygonMesh{T, U} <:QuadraticUnstructuredMesh2D{T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
    face_sets::Dict{String, BitSet}
end

function QuadraticPolygonMesh{T, U}(;
    name::String = "",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}} = SVector{6, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadraticPolygonMesh(name, points, faces, face_sets)
end

struct QuadraticTriangleMesh{T, U} <:QuadraticUnstructuredMesh2D{T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{SVector{6, U}}
    face_sets::Dict{String, BitSet}
end

function QuadraticTriangleMesh{T, U}(;
    name::String = "",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{SVector{6, U}} = SVector{6, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadraticTriangleMesh(name, points, faces, face_sets)
end
 
struct QuadraticQuadrilateralMesh{T, U} <:QuadraticUnstructuredMesh2D{T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{SVector{8, U}}
    face_sets::Dict{String, BitSet}
end

function QuadraticQuadrilateralMesh{T, U}(;
    name::String = "",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{SVector{8, U}} = SVector{8, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadraticQuadrilateralMesh(name, points, faces, face_sets)
end

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
