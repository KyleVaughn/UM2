export UnstructuredMesh
export points, name, groups, uint_cell_types

const VTK_TRIANGLE = 5
const VTK_QUAD = 9
const VTK_QUADRATIC_TRIANGLE = 22
const VTK_QUADRATIC_QUAD = 23
const VTK_TETRA = 10
const VTK_HEXAHEDRON = 12
const VTK_QUADRATIC_TETRA = 24
const VTK_QUADRATIC_HEXAHEDRON = 25

# Structure similar to Figure 8-35 of the VTK book.
struct UnstructuredMesh{Dim,T,U} <: AbstractMesh
    points::Vector{Point{Dim,T}}
    cell_array::Vector{U}
    cell_types::Vector{U}
    name::String
    groups::Dict{String,BitSet}
end

points(mesh::UnstructuredMesh) = mesh.points
name(mesh::UnstructuredMesh) = mesh.name
groups(mesh::UnstructuredMesh) = mesh.groups
uint_cell_types(mesh::UnstructuredMesh) = view(mesh.cell_types, 1:2:lastindex(mesh.cell_types))

function npoints(vtk_type::I) where {I<:Integer}
    if vtk_type == VTK_TRIANGLE
        return I(3)
    elseif vtk_type == VTK_QUAD
        return I(4)
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        return I(6)
    elseif vtk_type == VTK_QUADRATIC_QUAD
        return I(8)
    elseif vtk_type == VTK_TETRA
        return I(4)
    elseif vtk_type == VTK_HEXAHEDRON
        return I(8)
    elseif vtk_type == VTK_QUADRATIC_TETRA
        return I(10)
    elseif vtk_type == VTK_QUADRATIC_HEXAHEDRON
        return I(12)
    else
        error("Unsupported type.")
        return nothing
    end
end

function vtk_alias_string(vtk_type::I) where {I<:Integer}
    if vtk_type == VTK_TRIANGLE
        return "Triangle" 
    elseif vtk_type == VTK_QUAD
        return "Quadrilateral"
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        return "QuadraticTriangle"
    elseif vtk_type == VTK_QUADRATIC_QUAD
        return "QuadraticQuadrilateral"
    elseif vtk_type == VTK_TETRA
        return "Tetrahedron"
    elseif vtk_type == VTK_HEXAHEDRON
        return "Hexahedron"
    elseif vtk_type == VTK_QUADRATIC_TETRA
        return "QuadraticTetrahedron"
    elseif vtk_type == VTK_QUADRATIC_HEXAHEDRON
        return "QuadraticHexahedron"
    else
        error("Unsupported type.")
        return nothing
    end
end

function Base.show(io::IO, mesh::UnstructuredMesh{Dim,T,U}) where {Dim,T,U}
    println(io, "UnstructuredMesh{", Dim,", ", T,", ",U,"}")
    println(io, "  ├─ Name      : ", mesh.name)
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        println(io, "  ├─ Size (KB) : ", size_MB*1000)
    else
        println(io, "  ├─ Size (MB) : ", size_MB)
    end
    println(io, "  ├─ Points    : ", length(mesh.points))
    println(io, "  ├─ Cells     : ", length(mesh.cell_types) ÷ 2)
    types = uint_cell_types(mesh)
    unique_types = unique(types) 
    nunique_types = length(unique_types)
    for i = 1:nunique_types
        type = unique_types[i]
        ncells = count(x->x === type,  types)
        if i === nunique_types
            println(io, "  │  └─ ", rpad(vtk_alias_string(type), 22), ": ", ncells)
        else
            println(io, "  │  ├─ ", rpad(vtk_alias_string(type), 22), ": ", ncells)
        end
    end
    ngroups = length(mesh.groups)
    println(io, "  └─ Groups    : ", ngroups)
    if 0 < ngroups ≤ 5
        group_keys = sort!(collect(keys(mesh.groups)))
        for i = 1:ngroups
            if i === ngroups
                println(io, "     └─ ", group_keys[i])
            else
                println(io, "     ├─ ", group_keys[i])
            end
        end
    end
end
