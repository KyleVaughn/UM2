function Base.show(io::IO, mesh::VolumeMesh{D, T, U}) where {D, T, U}
    println(io, "VolumeMesh{", D, ", ", T, ", ", U, "}")
    println(io, "  ├─ Name      : ", mesh.name)
    size_B = sizeof(mesh)
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
