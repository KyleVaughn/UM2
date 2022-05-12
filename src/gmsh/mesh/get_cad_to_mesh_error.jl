export get_cad_to_mesh_error

function get_cad_to_mesh_error()
    @warn "Mass estimates from gmsh have a relative error of approximately 1%"
    errors = []
    for group in gmsh.model.get_physical_groups()
        gdim, gnum = group
        name = gmsh.model.get_physical_name(gdim, gnum)
        if startswith(name, "Material: ")
            tags = gmsh.model.get_entities_for_physical_group(gdim, gnum)
            mass = 0.0
            for tag in tags
                mass +=  gmsh.model.occ.get_mass(gdim, tag)
            end
            gmsh.plugin.setNumber("MeshVolume", "Dimension", gdim)
            gmsh.plugin.setNumber("MeshVolume", "PhysicalGroup",  gnum)
            vtag = gmsh.plugin.run("MeshVolume")
            data = gmsh.view.get_list_data(vtag)
            mesh_mass = data[3][1][end] 
            gmsh.view.remove(vtag)
            err_percent = 100*(mesh_mass - mass)/mass
            push!(errors, (name=name, cad_mass=mass, 
                           mesh_mass=mesh_mass, percent_error=err_percent))
            println(name)
            println("CAD Mass: $mass, Mesh Mass: $mesh_mass, Error: $err_percent %") 
        end
    end
    return errors 
end
