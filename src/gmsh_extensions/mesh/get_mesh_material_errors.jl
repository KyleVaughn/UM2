function get_mesh_material_errors()
    for group in gmsh.model.get_physical_groups()
        gdim, gnum = group
        name = gmsh.model.get_physical_name(gdim, gnum)
        if startswith(name, "Material: ")
            println(name)
            tags = gmsh.model.get_entities_for_physical_group(gdim, gnum)
            mass = 0.0
            for tag in tags
                mass +=  gmsh.model.occ.get_mass(gdim, tag)
            end
            println(mass)
            gmsh.plugin.setNumber("MeshVolume", "Dimension", gdim)
            gmsh.plugin.setNumber("MeshVolume", "PhysicalGroup",  gnum)
            vtag = gmsh.plugin.run("MeshVolume")
            data = gmsh.view.get_list_data(vtag)
            mesh_mass = data[3][1][end] 
            println(mesh_mass)
            gmsh.view.remove(vtag)
        end
    end
end
