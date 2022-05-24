export VolumeMesh
export points, name, groups, ishomogeneous, points_in_vtk_type 

struct VolumeMesh{Dim,T,U} <: AbstractMesh
    points::Vector{Point{Dim,T}}
    types::Vector{U}
    offsets::Vector{U}
    connectivity::Vector{U}
    name::String
    groups::Dict{String,BitSet}
end

points(mesh::VolumeMesh) = mesh.points
name(mesh::VolumeMesh) = mesh.name
groups(mesh::VolumeMesh) = mesh.groups

const VTK_TRIANGLE = 5
const VTK_QUAD = 9
const VTK_QUADRATIC_TRIANGLE = 22
const VTK_QUADRATIC_QUAD = 23
const VTK_TETRA = 10
const VTK_HEXAHEDRON = 12
const VTK_QUADRATIC_TETRA = 24
const VTK_QUADRATIC_HEXAHEDRON = 25

ishomogeneous(mesh::VolumeMesh) = all(i->mesh.types[1] === mesh.types[i], 2:length(mesh.types))

function points_in_vtk_type(vtk_type::I) where {I<:Integer}
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

function Base.show(io::IO, mesh::VolumeMesh{Dim,T,U}) where {Dim,T,U}
    println(io, "VolumeMesh{",Dim, ", ",T,", ",U,"}")
    println(io, "  ├─ Name      : ", mesh.name)
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        println(io, "  ├─ Size (KB) : ", size_MB*1000)
    else
        println(io, "  ├─ Size (MB) : ", size_MB)
    end
    println(io, "  ├─ Points    : ", length(mesh.points))
    if Dim === 3
        println(io, "  ├─ Faces     : ", length(mesh.offsets) - 1)
    else
        println(io, "  ├─ Cells     : ", length(mesh.offsets) - 1)
    end
    unique_types = unique(mesh.types)
    nunique_types = length(unique_types)
    for i = 1:nunique_types
        type = unique_types[i]
        nelements = count(x->x === type,  mesh.types)
        if i === nunique_types
            println(io, "  │  └─ ", rpad(vtk_alias_string(type), 22), ": ", nelements)
        else
            println(io, "  │  ├─ ", rpad(vtk_alias_string(type), 22), ": ", nelements)
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
