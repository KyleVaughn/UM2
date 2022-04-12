abstract type QuadraticPolyhedronMesh{T, U} <:QuadraticUnstructuredMesh3D{T, U} end

struct MixedQuadraticPolyhedronMesh{T, U} <:QuadraticPolyhedronMesh{T, U}
    name::String
    points::Vector{Point3D{T}}
    cells::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
    cell_sets::Dict{String, BitSet}
end

function MixedQuadraticPolyhedronMesh{T, U}(;
    name::String = "",
    points::Vector{Point3D{T}} = Point3D{T}[],
    cells::Vector{<:SArray{S, U, 1} where {S<:Tuple}} = SVector{10, U}[],
    cell_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return MixedQuadraticPolyhedronMesh(name, points, cells, cell_sets)
end

struct QuadraticTetrahedronMesh{T, U} <:QuadraticPolyhedronMesh{T, U}
    name::String
    points::Vector{Point3D{T}}
    cells::Vector{SVector{10, U}}
    cell_sets::Dict{String, BitSet}
end

function QuadraticTetrahedronMesh{T, U}(;
    name::String = "",
    points::Vector{Point3D{T}} = Point3D{T}[],
    cells::Vector{SVector{10, U}} = SVector{10, U}[],
    cell_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadraticTetrahedronMesh(name, points, cells, cell_sets)
end
 
struct QuadraticHexahedronMesh{T, U} <:QuadraticPolyhedronMesh{T, U}
    name::String
    points::Vector{Point3D{T}}
    cells::Vector{SVector{20, U}}
    cell_sets::Dict{String, BitSet}
end

function QuadraticHexahedronMesh{T, U}(;
    name::String = "",
    points::Vector{Point3D{T}} = Point3D{T}[],
    cells::Vector{SVector{20, U}} = SVector{20, U}[],
    cell_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return QuadraticHexahedronMesh(name, points, cells, cell_sets)
end

function Base.show(io::IO, mesh::MixedQuadraticPolyhedronMesh)
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
    println(io, "  ├─ Cells     : $(length(mesh.cells))")
    println(io, "  │  ├─ Tetrahedron    : $(count(x->x isa SVector{10},  mesh.cells))")
    println(io, "  │  └─ Hexahedron     : $(count(x->x isa SVector{20},  mesh.cells))")
    println(io, "  └─ Cell sets : $(length(keys(mesh.cell_sets)))")
end
