export VolumeMesh
export points, name, groups, materials, material_names, ishomogeneous, points_in_vtk_type 

struct VolumeMesh{Dim,T,U} <: AbstractMesh
    points::Vector{Point{Dim,T}}
    offsets::Vector{U}              # First index in connectivity
    connectivity::Vector{U}         # Point IDs that compose each element
    types::Vector{UInt8}            # VTK integer type of the element
    materials::Vector{UInt8}        # ID of the element's material
    material_names::Vector{String}
    name::String
    groups::Dict{String,BitSet}     # "Label"=>{IDs of elements with this label} 
end

points(mesh::VolumeMesh) = mesh.points
name(mesh::VolumeMesh) = mesh.name
groups(mesh::VolumeMesh) = mesh.groups
materials(mesh::VolumeMesh) = mesh.materials
material_names(mesh::VolumeMesh) = mesh.material_names

const VTK_TRIANGLE::UInt8 = 5
const VTK_QUAD::UInt8 = 9
const VTK_QUADRATIC_TRIANGLE::UInt8 = 22
const VTK_QUADRATIC_QUAD::UInt8 = 23
const VTK_TETRA::UInt8 = 10
const VTK_HEXAHEDRON::UInt8 = 12
const VTK_QUADRATIC_TETRA::UInt8 = 24
const VTK_QUADRATIC_HEXAHEDRON::UInt8 = 25

ishomogeneous(mesh::VolumeMesh) = all(i->mesh.types[1] === mesh.types[i], 2:length(mesh.types))

function points_in_vtk_type(vtk_type::I) where {I<:Integer}
    if vtk_type == VTK_TRIANGLE
        return 3
    elseif vtk_type == VTK_QUAD
        return 4
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        return 6
    elseif vtk_type == VTK_QUADRATIC_QUAD
        return 8
    elseif vtk_type == VTK_TETRA
        return 4
    elseif vtk_type == VTK_HEXAHEDRON
        return 8
    elseif vtk_type == VTK_QUADRATIC_TETRA
        return 10
    elseif vtk_type == VTK_QUADRATIC_HEXAHEDRON
        return 12
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
    size_B = Base.summarysize(mesh)
    if size_B < 1e6
        println(io, "  ├─ Size (KB) : ", string(@sprintf("%.3f", size_B/1000)))
    else
        println(io, "  ├─ Size (MB) : ", string(@sprintf("%.3f",size_B/1e6)))
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
    println(io, "  ├─ Materials : ", length(mesh.material_names))
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
