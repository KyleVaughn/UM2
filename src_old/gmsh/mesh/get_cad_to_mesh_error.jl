export get_cad_to_mesh_error

function get_cad_to_mesh_error()
    errors = NamedTuple{(:name, :cad_mass, :mesh_mass, :percent_error),
                        Tuple{String, Float64, Float64, Float64}}[]
    for group in gmsh.model.get_physical_groups()
        gdim, gnum = group
        name = gmsh.model.get_physical_name(gdim, gnum)
        if startswith(name, "Material: ")
            tags = gmsh.model.get_entities_for_physical_group(gdim, gnum)
            mass = 0.0
            for tag in tags
                mass += gmsh.model.occ.get_mass(gdim, tag)
            end
            gmsh.plugin.setNumber("MeshVolume", "Dimension", gdim)
            gmsh.plugin.setNumber("MeshVolume", "PhysicalGroup", gnum)
            vtag = gmsh.plugin.run("MeshVolume")
            data = gmsh.view.get_list_data(vtag)
            mesh_mass = data[3][1][end]
            gmsh.view.remove(vtag)
            err_percent = 100 * (mesh_mass - mass) / mass
            push!(errors,
                  (name = name, cad_mass = mass,
                   mesh_mass = mesh_mass, percent_error = err_percent))
        end
    end
    return errors
end

function Base.show(io::IO,
                   err::NamedTuple{(:name, :cad_mass, :mesh_mass, :percent_error),
                                   Tuple{String, Float64, Float64, Float64}})
    print(io, err.name)
    print(io, ", Error (%): " * string(@sprintf("%.4f", err.percent_error)))
    print(io, ", CAD Mass: " * string(@sprintf("%.4f", err.cad_mass)))
    return print(io, ", Mesh Mass: " * string(@sprintf("%.4f", err.mesh_mass)))
end
