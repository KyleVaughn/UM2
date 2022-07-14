export set_mesh_field_using_materials

function set_mesh_field_using_materials(materials::Vector{Material})
    @info "Setting mesh size field using materials"
    # Setup a field with a constant mesh size (material.lc) inside the 
    # entities that make up that material
    field_ids = Vector{Int32}(undef, length(materials))
    for (i, material) in enumerate(materials)
        fid = gmsh.model.mesh.field.add("Constant")
        gmsh.model.mesh.field.set_number(fid, "VIn", material.lc)
        field_ids[i] = fid
    end
    # Populate each of the fields with the highest dimensional entities in the material
    # physical group
    material_dict = Dict{String, Vector{Tuple{Int32, Int32}}}()
    material_names = ["Material: " * mat.name for mat in materials]
    for name in material_names
        material_dict[name] = Tuple{Int32, Int32}[]
    end
    for group in gmsh.model.get_physical_groups()
        gdim, gnum = group
        name = gmsh.model.get_physical_name(gdim, gnum)
        if startswith(name, "Material: ")
            tags = gmsh.model.get_entities_for_physical_group(gdim, gnum)
            dtags = [(gdim, tag) for tag in tags]
            # Empty vector, or these ents are higher dim
            if length(material_dict[name]) == 0 || gdim > material_dict[name][1][1]
                material_dict[name] = dtags
            end
        end
    end
    # Assign the entities to the fields
    for (mat_id, fid) in enumerate(field_ids)
        name = material_names[mat_id]
        dtags = material_dict[name]
        edim = dtags[1][1]
        tags = [dt[2] for dt in dtags]
        if edim == 0
            gmsh.model.mesh.field.set_numbers(fid, "PointsList", tags)
        elseif edim == 1
            gmsh.model.mesh.field.set_numbers(fid, "CurvesList", tags)
        elseif edim == 2
            gmsh.model.mesh.field.set_numbers(fid, "SurfacesList", tags)
        else # edim == 3
            gmsh.model.mesh.field.set_numbers(fid, "VolumesList", tags)
        end
    end
    # Create a field that takes the min of each and set as background mesh
    fid = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.set_numbers(fid, "FieldsList", field_ids)
    gmsh.model.mesh.field.set_as_background_mesh(fid)
    gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
    return field_ids
end
