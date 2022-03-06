struct ConvexPolyhedronMesh{T, U} <:LinearUnstructuredMesh3D{T, U}
    name::String
    points::Vector{Point3D{T}}
    cells::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
    cell_sets::Dict{String, BitSet}
end

function ConvexPolyhedronMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point3D{T}} = Point3D{T}[],
    cells::Vector{<:SArray{S, U, 1} where {S<:Tuple}} = SVector{4, U}[],
    cell_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return ConvexPolyhedronMesh(name, points, cells, cell_sets)
end

struct TetrahedronMesh{T, U} <:LinearUnstructuredMesh3D{T, U}
    name::String
    points::Vector{Point3D{T}}
    cells::Vector{SVector{4, U}}
    cell_sets::Dict{String, BitSet}
end

function TetrahedronMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point3D{T}} = Point3D{T}[],
    cells::Vector{SVector{4, U}} = SVector{4, U}[],
    cell_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return TetrahedronMesh(name, points, cells, cell_sets)
end

struct HexahedronMesh{T, U} <:LinearUnstructuredMesh3D{T, U}
    name::String
    points::Vector{Point3D{T}}
    cells::Vector{SVector{8, U}}
    cell_sets::Dict{String, BitSet}
end

function HexahedronMesh{T, U}(;
    name::String = "default_name",
    points::Vector{Point3D{T}} = Point3D{T}[],
    cells::Vector{SVector{8, U}} = SVector{8, U}[],
    cell_sets::Dict{String, BitSet} = Dict{String, BitSet}()
    ) where {T, U}
    return HexahedronMesh(name, points, cells, cell_sets)
end

function Base.show(io::IO, mesh::ConvexPolyhedronMesh)
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
    println(io, "  │  ├─ Tetrahedron    : $(count(x->x isa SVector{4},  mesh.cells))")
    println(io, "  │  └─ Hexahedron     : $(count(x->x isa SVector{8},  mesh.cells))")
    println(io, "  └─ Cell sets : $(length(keys(mesh.cell_sets)))")
end
