struct ConvexPolygonMesh{T, U} <:LinearUnstructuredMesh2D{T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
    face_sets::Dict{String, BitSet}
end

function ConvexPolygonMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}} = SVector{3, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return ConvexPolygonMesh(name, points, faces, face_sets)
end

struct TriangleMesh{T, U} <:LinearUnstructuredMesh2D{T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{SVector{3, U}}
    face_sets::Dict{String, BitSet}
end

function TriangleMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{SVector{3, U}} = SVector{3, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return TriangleMesh(name, points, faces, face_sets)
end

struct QuadrilateralMesh{T, U} <:LinearUnstructuredMesh2D{T, U}
    name::String
    points::Vector{Point2D{T}}
    faces::Vector{SVector{4, U}}
    face_sets::Dict{String, BitSet}
end

function QuadrilateralMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point2D{T}} = Point2D{T}[],
    faces::Vector{SVector{4, U}} = SVector{4, U}[],
    face_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadrilateralMesh(name, points, faces, face_sets)
end

function Base.show(io::IO, mesh::ConvexPolygonMesh)
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
    println(io, "  │  ├─ Triangle       : $(count(x->x isa SVector{3},  mesh.faces))")
    println(io, "  │  └─ Quadrilateral  : $(count(x->x isa SVector{4},  mesh.faces))")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end

function check_convexity(mesh::ConvexPolygonMesh)

end
