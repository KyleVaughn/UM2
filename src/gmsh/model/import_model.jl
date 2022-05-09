export import_model

"""
    function import_model(path::String; names::Bool=false)

Import a CAD model into gmsh. `path` is the path to the CAD file. `names` is an optional
argument, which when set to `true` will add any named CAD groups to the model's physical groups.

Returns a Vector{Material} containing material information that was gathered from the file.
"""
function import_model(path::String; names::Bool=false)
    @info "Importing '"*path*"'"
    if !Bool(gmsh.is_initialized())
        gmsh.initialize()
    end

    gmsh.merge(path)
    # OCC converts everything to mm. We want cm, so we shrink by 1//10
    color_dict = get_entities_by_color()
    dim_tags = gmsh.model.get_entities()
    gmsh.model.occ.dilate(dim_tags, 0, 0, 0, 1//10, 1//10, 1//10)
    gmsh.model.occ.synchronize()
    # Reassign colors that we lost in dilation
    for color in keys(color_dict)
        r,g,b,a = color
        gmsh.model.set_color(color_dict[color], r, g, b, a) 
    end

    if names
        add_cad_names_to_physical_groups()
    end

    upath = uppercase(path)
    if endswith(upath, ".STEP") || endswith(upath, ".STP")
        return add_materials_from_step(path)
    else
        return Material[]
    end
end

function add_materials_from_step(path::String)
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
            r = convert(N0f8, parse(Float64, splitline[2]))
            g = convert(N0f8, parse(Float64, splitline[3]))
            b = convert(N0f8, parse(Float64, splitline[4][1:end-2]))
            color = RGBA(r, g, b, N0f8(1))
            mat = Material(name, color, 1.0)
            if mat ∉ materials
                push!(materials, mat)
            end
        end
    end
    close(file)
    add_materials_to_physical_groups_by_color(materials)
    return materials
end
