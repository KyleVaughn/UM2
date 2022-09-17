vtk_type(::Type{<:Hexahedron})             = VTK_HEXAHEDRON
vtk_type(::Type{<:QuadraticTetrahedron})   = VTK_QUADRATIC_TETRA
vtk_type(::Type{<:QuadraticHexahedron})    = VTK_QUADRATIC_HEXAHEDRON

function Base.show(io::IO, mesh::PolytopeVertexMesh{D, T, P}) where {D, T, P}
    print(io, "PolytopeVertexMesh{", D, ", ", T, ", ")
    if isconcretetype(P)
        println(io, alias_string(P), "{", vertextype(P), "}}")
    else
        println(io, P, "}")
    end
    println(io, "  ├─ Name      : ", mesh.name)
    size_B = sizeof(mesh)
    if size_B < 1e6
        println(io, "  ├─ Size (KB) : ", string(@sprintf("%.3f", size_B/1000)))
    else
        println(io, "  ├─ Size (MB) : ", string(@sprintf("%.3f", size_B/1e6)))
    end
    println(io, "  ├─ Vertices  : ", length(mesh.vertices))
    println(io, "  ├─ Polytopes : ", length(mesh.polytopes))
    poly_types = unique!(map(typeof, mesh.polytopes))
    npoly_types = length(poly_types)
    for i in 1:npoly_types
        poly_type = poly_types[i]
        npoly = count(x -> x isa poly_type, mesh.polytopes)
        if i === npoly_types
            println(io, "  │  └─ ", rpad(alias_string(poly_type), 22), ": ", npoly)
        else
            println(io, "  │  ├─ ", rpad(alias_string(poly_type), 22), ": ", npoly)
        end
    end
    println(io, "  ├─ Materials : ", length(mesh.material_names))
    ngroups = length(mesh.groups)
    println(io, "  └─ Groups    : ", ngroups)
    if 0 < ngroups ≤ 5
        group_keys = sort(collect(keys(mesh.groups)))
        for i in 1:ngroups
            if i === ngroups
                println(io, "     └─ ", group_keys[i])
            else
                println(io, "     ├─ ", group_keys[i])
            end
        end
    end
end
