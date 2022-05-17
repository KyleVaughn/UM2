# IO routines for the Abaqus .inp file format
const supported_abaqus_element_types = (
    # triangle
    "CPS3",
    # quadratic triangle
    "CPS6",
    # quadrilateral
    "CPS4", 
    # quadratic quadrilateral
    "CPS8",
    # tetrahedron
    "C3D4",
    # hexahedron
    "C3D8",
    # quadratic tetrahedron
    "C3D10",
    # quadratic hexahedron"
    "C3D20"
   )

# TODO: 
#   1) Read non 1:N nodes/elements and map them to the appropriate 1:N indices
#   2) Read all element types and discard the unused lower dimensional elements
#   3) Figure out if pushing to a vector every line, or if reading the file twice (once to
#       determine vector size, and once to populate the vector) is faster for reading 
#       nodes or elements
function read_abaqus(path::String, ::Type{T}) where {T<:AbstractFloat}
    file = open(path, "r")
    try
        name = "default_name"
        element_vecs = Vector{UInt64}[] 
        element_sets = Dict{String, BitSet}()
        points = Point{3,T}[]
        is2D = false
        is3D = false
        while !eof(file)
            line = readline(file)
            if length(line) > 0
                if startswith(line, "**") # Comment
                    continue
                elseif "*Heading" == line
                    name = String(strip(readline(file)))
                    if occursin(".inp", name)
                        name = name[1:length(name)-4]
                    end
                elseif "*NODE" == line
                    _read_abaqus_nodes!(file, points)
                elseif occursin("*ELEMENT", line)
                    splitline = split(line)
                    element_type = String(strip(replace(splitline[2], ("type=" => "")), ','))
                    if element_type ∉ supported_abaqus_element_types
                        error("$element_type is not a supported abaqus element type")
                    end
                    if startswith(element_type, "CP")
                        is2D = true
                    else
                        is3D = true
                    end
                    _read_abaqus_elements!(file, element_vecs, element_type)
                elseif occursin("*ELSET", line)
                    splitline = split(line)
                    set_name = String(replace(splitline[1], ("*ELSET,ELSET=" => "")))
                    element_sets[set_name] = _read_abaqus_elset(file)
                end
            end
        end
        if is2D && is3D
            error("File contains both surface (CPS) and volume (C3) elements."*
                  "Limit element types to be CPS or C3, so mesh dimension is inferrable") 
        end
        return _create_mesh_from_elements(is3D, name, points, element_vecs, element_sets)
    finally
        close(file)
    end
    return nothing
end

function _read_abaqus_nodes!(file::IOStream, points::Vector{Point{3,T}}) where {T}
    # Count the number of nodes
    file_position = position(file)
    npoints = 0
    line = readline(file) 
    while !('*' == line[1])
        npoints += 1
        line = readline(file)
    end
    seek(file, file_position)
    # Allocate and populate a vector of points
    new_points = Vector{Point{3,T}}(undef, npoints)
    line = readline(file)
    ipt = 0
    while !('*' == line[1])
        ipt += 1
        xyz = parse.(T, strip.(view(split(line),2:4), [',']))
        new_points[ipt] = Point{3,T}(xyz[1], xyz[2], xyz[3])
        file_position = position(file)
        line = readline(file)
    end
    seek(file, file_position)
    append!(points, new_points)
    return nothing
end

function _read_abaqus_elements!(file::IOStream, elements::Vector{Vector{UInt64}}, 
                                element_type::String)
    line = readline(file)
    file_position = position(file)
    while !('*' == line[1] || eof(file))
        splitline = split(line)
        vertex_ids = parse.(UInt64, strip.(view(splitline, 2:length(splitline)), [',']))
        push!(elements, vertex_ids)
        file_position = position(file)
        line = readline(file)
    end
    seek(file, file_position)
    return elements
end

function _read_abaqus_elset(file::IOStream)
    splitline = strip.(split(readuntil(file, "*")), [','])
    seek(file, position(file)-1)
    return BitSet(parse.(Int64, splitline))
end
