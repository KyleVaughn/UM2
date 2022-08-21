export AbaqusFile

export get_material_names

# ABAQUS FILE
# -----------------------------------------------------------------------------    
#    
# An intermediate representation of an Abaqus input file and a mesh.
# 

struct AbaqusFile{T <: AbstractFloat, I <: Integer}
    filepath::String
    name::String
    nodes::Vector{Point3{T}}
    element_types::Vector{Int8}
    element_offsets::Vector{I}
    elements::Vector{I}
    elsets::Dict{String, Set{I}}
end

# -- IO --

function parse_nodes!(file::IOStream, 
                      nodes::Vector{Point3{T}}
                     ) where {T <: AbstractFloat, I <: Integer}
    line = ""
    while !eof(file)
        line = readline(file)
        if line[1] == '*'
            break
        end
        words = split(line, ',')
        x = parse(T, words[2])
        y = parse(T, words[3])
        z = parse(T, words[4])
        push!(nodes, Point3(x, y, z))
    end
    return line
end

function parse_elements!(file::IOStream, 
                         line::String,
                         element_types::Vector{Int8},
                         element_offsets::Vector{I},
                         elements::Vector{I},
                         total_offset::I
                        ) where {T <: AbstractFloat, I <: Integer}
    offset = I(line[19] - 48)
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
            push!(elements, parse(I, words[i + 1]))
        end
    end
    return (line, total_offset)
end

function parse_elsets!(file::IOStream, 
                       line::String,
                       elsets::Dict{String, Set{I}}) where {I <: Integer}
    elset_name = line[14:end]
    elset = Set{I}()
    while !eof(file)
        line = readline(file)
        if line[1] == '*'
            break
        end
        words = split(line, ',')
        for i in 1:length(words)
            if 0 < length(words[i])
                push!(elset, parse(I, words[i]))
            end
        end
    end
    elsets[elset_name] = elset
    return line
end

# -- Constructors --

function AbaqusFile(filepath::String)
    T = UM2_MESH_FLOAT_TYPE 
    I = UM2_MESH_INT_TYPE
    name = ""
    nodes = Point3{T}[]
    element_types = Int8[]
    element_offsets = I[]
    elements = I[]
    elsets = Dict{String, Set{I}}()
    total_offset = I(1)
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
    return AbaqusFile{T, I}(filepath, name, nodes, element_types, 
                            element_offsets, elements, elsets)
end

# -- Getters --

function get_material_names(AbaqusFile)
    material_names = String[]
    for elset_name in keys(AbaqusFile.elsets)
        if startswith(elset_name, "Material:")
            push!(material_names, elset_name)
        end
    end
    sort!(material_names)
    return material_names
end
