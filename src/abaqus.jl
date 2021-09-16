const abaqus_to_vtk_type = Dict(
    # 2D
    # triangle
    "CPS3"  => 5,
    "STRI3" => 5,
    # triangle6
    "CPS6" => 22,
    # quadrilateral
    "CPS4" => 9,
    # quad8
    "CPS8" => 23,
    "M3D9" => 23
   )

function read_abaqus_2d(filepath::String; float_type=Float64)
    @info "Reading $filepath"
    # NOTE: There is a crucial assumption here that elements and nodes are listed 1 to N,
    # not 8, 10, 9 or anything funky/out of order.
    name = "DefaultMeshName"
    file = open(filepath, "r")
    faces = Union{
                   NTuple{4, Int64},
                   NTuple{5, Int64},
                   NTuple{7, Int64},
                   NTuple{9, Int64}
                  }[]
    face_sets = Dict{String, Set{Int64}}()
    points = Point_2D{float_type}[]
    while !eof(file)
        line_split = split(readline(file))
        if length(line_split) > 0
            if occursin("**", line_split[1]) # Comment
                continue
            elseif occursin("*Heading", line_split[1])
                name = strip(readline(file))
                if occursin(".inp", name)
                    name = name[1:length(name)-4]
                end
                name = String(name)
            elseif occursin("*NODE", line_split[1])
                points = _read_abaqus_nodes_2d(file, float_type)
            elseif occursin("*ELEMENT", line_split[1])
                element_type = String(strip(replace(line_split[2], ("type=" => "")), ','))
                faces = vcat(faces, _read_abaqus_elements(file, element_type))
            elseif occursin("*ELSET", line_split[1])
                set_name = String(replace(line_split[1], ("*ELSET,ELSET=" => "")))
                face_sets[set_name] = _read_abaqus_elset(file)
            end
        end
    end
    close(file)
    return UnstructuredMesh_2D(points = points,
                               faces = faces,
                               name = name,
                               face_sets = face_sets
                              )
end

function _read_abaqus_nodes_2d(file::IOStream, type::Type{T}) where {T <: AbstractFloat}
    points = Point_2D{type}[]
    line_split = strip.(split(readline(file)), [','])
    line_position = position(file)
    while !occursin("*", line_split[1])
        xyz = parse.(type, line_split[2:4])
        push!(points, Point_2D(xyz[1], xyz[2]))
        line_position = position(file)
        line_split = strip.(split(readline(file)), [','])
    end
    seek(file, line_position)
    return points
end

function _read_abaqus_elements(file::IOStream, element_type::String)
    if !(element_type ∈  keys(abaqus_to_vtk_type))
        error("$element_type is not in the abaqus to vtk type conversion dictionary")
    end
    type = abaqus_to_vtk_type[element_type]
    faces = Union{
                   NTuple{4, Int64},
                   NTuple{5, Int64},
                   NTuple{7, Int64},
                   NTuple{9, Int64}
                  }[]
    line_split = strip.(split(readline(file)), [','])
    line_position = position(file)
    while !occursin("*", line_split[1])
        vertex_IDs = parse.(Int64, line_split[2:length(line_split)])
        if length(vertex_IDs) == 9
            push!(faces, Tuple(vcat(type, vertex_IDs[1:8])))
        else
            push!(faces, Tuple(vcat(type, vertex_IDs)))
        end
        line_position = position(file)
        line_split = strip.(split(readline(file)), [','])
    end
    seek(file, line_position) 
    return faces
end

function _read_abaqus_elset(file::IOStream)
    line_split = strip.(split(readuntil(file, "*")), [','])
    seek(file, position(file)-1)
    return Set(parse.(Int64, line_split))
end
