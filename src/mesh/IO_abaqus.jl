# IO routines for the Abaqus .inp file format
const valid_abaqus_element_types = (
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

function read_abaqus(filepath::String, floattype::Type{T}=Float64) where {T<:AbstractFloat}
    @info "Reading "*filepath
    # NOTE: There is a critical assumption here that elements and nodes are listed 1 to N,
    # not 8, 10, 9 or anything funky/out of order.
    file = open(filepath, "r")
    name = "default_name"
    element_vecs = Vector{UInt64}[] 
    element_sets = Dict{String, BitSet}()
    points = Point3D{floattype}[]
    is3D = false
    while !eof(file)
        line = readline(file)
        if length(line) > 0
            if "**" == @view line[1:2] # Comment
                continue
            elseif "*Heading" == line
                name = String(strip(readline(file)))
                if occursin(".inp", name)
                    name = name[1:length(name)-4]
                end
            elseif "*NODE" == line
                _read_abaqus_nodes!(file, points)
            elseif occursin("*ELEMENT", line)
                linesplit = split(line)
                element_type = String(strip(replace(linesplit[2], ("type=" => "")), ','))
                if element_type[1:2] == "C3"
                    is2D = true
                end
                _read_abaqus_elements!(file, elements_vecs, element_type)
            elseif occursin("*ELSET", line)
                linesplit = split(line)
                set_name = String(replace(linesplit[1], ("*ELSET,ELSET=" => "")))
                element_sets[set_name] = _read_abaqus_elset(file)
            end
        end
    end
    close(file)
    return _create_mesh_from_elements(is3D, name, points, elements_vecs, element_sets)
end

function _read_abaqus_nodes!(file::IOStream, points::Vector{Point3D{T}}) where {T}
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
    new_points = Vector{Point3D{T}}(undef, npoints)
    line = readline(file)
    ipt = 0
    while !('*' == line[1])
        ipt += 1
        xyz = parse.(T, strip.(view(split(line),2:4), [',']))
        new_points[ipt] = Point(xyz[1], xyz[2], xyz[3])
        file_position = position(file)
        line = readline(file)
    end
    seek(file, file_position)
    append!(points, new_points)
    return nothing
end

function _read_abaqus_elements!(file::IOStream, elements::Vector{Vector{UInt64}}, 
                                element_type::String)
    if !(element_type ∈ valid_abaqus_types)  
        error("$element_type is not in the valid abaqus types")
    end
    line = readline(file)
    file_position = position(file)
    while !('*' == line[1] || eof(file))
        linesplit = split(line)
        vertex_ids = parse.(UInt64, strip.(view(linesplit, 2:length(linesplit)), [',']))
        push!(elements, vertex_ids)
        file_position = position(file)
        line = readline(file)
    end
    seek(file, file_position)
    return faces
end

function _read_abaqus_elset(file::IOStream)
    linesplit = strip.(split(readuntil(file, "*")), [','])
    seek(file, position(file)-1)
    return BitSet(parse.(Int64, linesplit))
end
