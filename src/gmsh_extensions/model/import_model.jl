function import_model(path::String; names::Bool=false)
    if !Bool(gmsh.is_initialized())
        gmsh.initialize()
        gmsh.option.set_number("General.Verbosity", 2)
    end

    gmsh.merge(path)

    if names
        add_cad_entity_names_to_physical_groups(2)
    end

    upath = uppercase(path)
    if endswith(upath, ".STEP") || endswith(upath, ".STP")
        return get_materials_from_step(path)
    else
        return Material[]
    end
end

function get_materials_from_step(path::String)
    file = open(path, "r")
    materials = Material[]
    while !eof(file)
        line = readline(file)
        if occursin("COLOUR_RGB", line)
            splitline = split(line, ",")
            # Account for line continuation
            if splitline[3] == "" || splitline[4] == ""
                line2 = readline(file)
                line = line*line2
                splitline = split(line, ",")
            end
            name = String(split(splitline[1], "'")[2])
            r = Int32(round(255*parse(Float64, splitline[2])))
            g = Int32(round(255*parse(Float64, splitline[3])))
            b = Int32(round(255*parse(Float64, splitline[4][1:end-2])))
            mat = Material(name, (r, g, b, Int32(255)), 1.0)
            if mat ∉ materials
                push!(materials, mat)
            end
        end
    end
    close(file)
    color_to_ent = get_entities_by_color(2)
    add_materials_to_physical_groups_by_color(materials, color_to_ent, 2)
    return materials
end
