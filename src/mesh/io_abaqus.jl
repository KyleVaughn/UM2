# ABAQUS FILE FORMAT

# TODO: Make this more efficient. See C++ version for ideas.

# -- Read -- 

function parse_nodes!(file::IOStream, 
                      nodes::Vector{Point2{UM_F}})
    line = ""
    while !eof(file)
        line = readline(file)
        if line[1] == '*'
            break
        end
        words = split(line, ',')
        x = parse(UM_F, words[2])
        y = parse(UM_F, words[3])
        push!(nodes, Point2(x, y))
    end
    return line
end

function parse_elements!(file::IOStream, 
                         line::String,
                         element_types::Vector{Int8},
                         element_offsets::Vector{UM_I},
                         elements::Vector{UM_I},
                         total_offset::UM_I)
    offset = UM_I(line[19] - 48)
    abaqus_type = Int8(0)
    if offset == 3
        abaqus_type = ABAQUS_CPS3
    elseif offset == 4
        abaqus_type = ABAQUS_CPS4
    elseif offset == 6
        abaqus_type = ABAQUS_CPS6
    elseif offset == 8
        abaqus_type = ABAQUS_CPS8
    else
        error("Unsupported element type: " * string(offset))
    end
    while !eof(file)
        line = readline(file)
        if line[1] == '*'
            break
        end
        push!(element_types, abaqus_type)
        push!(element_offsets, total_offset)
        total_offset += offset
        words = split(line, ',')
        for i in 1:offset
            push!(elements, parse(UM_I, words[i + 1]))
        end
    end
    return (line, total_offset)
end

function parse_elsets!(file::IOStream, 
                       line::String,
                       elsets::Dict{String, Set{UM_I}})
    elset_name = line[14:end]
    elset = Set{UM_I}()
    while !eof(file)
        line = readline(file)
        if line[1] == '*'
            break
        end
        words = split(line, ',')
        for i in 1:length(words)
            lsword = lstrip(words[i])
            if 0 < length(lsword)
                push!(elset, parse(UM_I, lsword))
            end
        end
    end
    elsets[elset_name] = elset
    return line
end

function read_abaqus_file(filepath::String)
    @info "Reading ABAQUS file: '" * filepath * "'"
    name = ""
    nodes = Point2{UM_F}[]
    element_types = Int8[]
    element_offsets = UM_I[]
    elements = UM_I[]
    elsets = Dict{String, Set{UM_I}}()
    total_offset = UM_I(1)
    line = ""
    loop_again = false
    file = open(filepath, "r")
    try
        while loop_again || !eof(file)
            if loop_again
                loop_again = false
            else
                line = readline(file)
            end
            if startswith(line, "**")
                continue
            elseif startswith(line, "*Heading")
                line = readline(file)
                name = line[2:end]
                if endswith(name, ".inp")
                    name = name[1:end-4]
                end
            elseif startswith(line, "*NODE")
                line = parse_nodes!(file, nodes)
                loop_again = true
            elseif startswith(line, "*ELEMENT")
                (line, total_offset) = parse_elements!(file, line, element_types, 
                                                       element_offsets, elements,
                                                       total_offset)
                loop_again = true
            elseif startswith(line, "*ELSET")
                line = parse_elsets!(file, line, elsets)
                loop_again = true
            end
        end
    catch
        error("Error while reading: " * filepath)
    finally
        close(file)
    end
    push!(element_offsets, total_offset)
    # Material names
    material_names = String[]
    for elset_name in keys(elsets)
        if startswith(elset_name, "Material:")
            push!(material_names, elset_name[11:end])
        end
    end
    sort!(material_names)
    nfaces = length(element_types)
    materials = zeros(Int8, nfaces)
    for (i, mat_name) in enumerate(material_names)
        for id in elsets[mat_name]
            if materials[id] != 0
                error("Element " * string(id) * " already has a material")
            end
            materials[id] = i 
        end
    end

    return material_names, materials, 
           MeshFile(filepath, ABAQUS_FORMAT, name, nodes, element_types, 
                                      element_offsets, elements, elsets)
end
