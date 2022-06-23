export VolumeMesh
export points, name, groups, materials, material_names, nelements, ishomogeneous,
       islinear, isquadratic

struct VolumeMesh{D, T, U} <: AbstractMesh
    points::Vector{Point{D, T}}
    offsets::Vector{U}
    connectivity::Vector{U}         # Point IDs that compose each element
    materials::Vector{UInt8}        # ID of the element's material
    material_names::Vector{String}
    name::String
    groups::Dict{String, BitSet}    # "Label"=>{IDs of elements with this label} 
end

points(mesh::VolumeMesh) = mesh.points
name(mesh::VolumeMesh) = mesh.name
groups(mesh::VolumeMesh) = mesh.groups
materials(mesh::VolumeMesh) = mesh.materials
material_names(mesh::VolumeMesh) = mesh.material_names

nelements(mesh::VolumeMesh) = length(mesh.offsets) - 1
offset_diff(i::Integer, mesh::VolumeMesh) = mesh.offsets[i + 1] - mesh.offsets[i]

function typeof_face(i::Integer, mesh::VolumeMesh{2, T, U}) where {T, U}
    npt = offset_diff(i, mesh)
    if npt == 3
        return Triangle{U}
    elseif npt == 4
        return Quadrilateral{U}
    elseif npt == 6
        return QuadraticTriangle{U}
    elseif npt == 8
        return QuadraticQuadrilateral{U}
    else
        error("Invalid number of points.")
    end
end

function ishomogeneous(mesh::VolumeMesh)
    # This can be fooled if there are 3 types of elements
    return mod(length(mesh.connectivity), nelements(mesh)) == 0
end

function islinear(mesh::VolumeMesh{2})
    # This can be fooled by mixing linear and quadratic elements, which
    # is not supported
    return length(mesh.connectivity) / nelements(mesh) ≤ 4.000001
end
isquadratic(mesh::VolumeMesh{2}) = !islinear(mesh)

function _volume_mesh_points_to_vtk_type(dim::Integer, npt::Integer)
    if dim == 2
        if npt == 3
            return VTK_TRIANGLE
        elseif npt == 4
            return VTK_QUAD
        elseif npt == 6
            return VTK_QUADRATIC_TRIANGLE
        elseif npt == 8
            return VTK_QUADRATIC_QUAD
        else
            error("Invalid number of points.")
        end
    elseif dim == 3
        error("Not implemented yet.")
    else
        error("Invalid dimension.")
    end
end

function Base.show(io::IO, mesh::VolumeMesh{D, T, U}) where {D, T, U}
    println(io, "VolumeMesh{", D, ", ", T, ", ", U, "}")
    println(io, "  ├─ Name      : ", mesh.name)
    size_B = Base.summarysize(mesh)
    if size_B < 1e6
        println(io, "  ├─ Size (KB) : ", string(@sprintf("%.3f", size_B/1000)))
    else
        println(io, "  ├─ Size (MB) : ", string(@sprintf("%.3f", size_B/1e6)))
    end
    println(io, "  ├─ Points    : ", length(mesh.points))
    nel = nelements(mesh)
    if D === 2
        println(io, "  ├─ Faces     : ", nel)
    else
        println(io, "  ├─ Cells     : ", nel)
    end
    npt = [mesh.offsets[i + 1] - mesh.offsets[i] for i in 1:nel]
    unique_npt = unique(npt)
    nunique_npt = length(unique_npt)
    for i in 1:nunique_npt
        npt = unique_npt[i]
        nelements = count(x -> x === npt, npt)
        vtk_alias = vtk_alias_string(_volume_mesh_points_to_vtk_type(D, npt))
        if i === nunique_npt
            println(io, "  │  └─ ", rpad(vtk_alias, 22), ": ", nel)
        else
            println(io, "  │  ├─ ", rpad(vtk_alias, 22), ": ", nel)
        end
    end
    println(io, "  ├─ Materials : ", length(mesh.material_names))
    ngroups = length(mesh.groups)
    println(io, "  └─ Groups    : ", ngroups)
    if 0 < ngroups ≤ 5
        group_keys = sort!(collect(keys(mesh.groups)))
        for i in 1:ngroups
            if i === ngroups
                println(io, "     └─ ", group_keys[i])
            else
                println(io, "     ├─ ", group_keys[i])
            end
        end
    end
end
