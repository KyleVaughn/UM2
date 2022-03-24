function apply_material_colors(materials::Vector{Material})
    material_names = ["Material: "*mat.name for mat in materials]
    for group in gmsh.model.get_physical_groups()
        gdim, gnum = group
        tags = gmsh.model.get_entities_for_physical_group(gdim, gnum)
        name = gmsh.model.get_physical_name(gdim, gnum)
        if startswith(name, "Material: ")
            mat_id = findfirst(mat_name->name == mat_name, material_names)
            r,g,b,a = materials[mat_id].color
            dtags = [(gdim, tag) for tag in tags]
            gmsh.model.set_color(dtags, r, g, b, a)  
        end
    end
    return nothing
end
