const abaqus_to_vtk_type = Dict{String, UInt64}(
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
    faces = Vector{UInt64}[]
    face_sets = Dict{String, Set{UInt64}}()
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
                append!(faces, _read_abaqus_elements(file, element_type))
            elseif occursin("*ELSET", line_split[1])
                set_name = String(replace(line_split[1], ("*ELSET,ELSET=" => "")))
                face_sets[set_name] = _read_abaqus_elset(file)
            end
        end
    end
    close(file)
    # We can save a lot on memory and boost performance by choosing the smallest usable
    # integer for our mesh. The relationship between faces, edges, and vertices is
    # Triangle
    #   F ≈ 2V
    #   E ≈ 3F/2
    # Quadrilateral
    #   F ≈ V
    #   E ≈ 2F
    # Triangle6
    #   2F ≈ V
    #   E ≈ 3F/2
    # Quadrilateral8
    #   4F ≈ V 
    #   E ≈ 2F
    # We see that V ≤ F ≤ 2E, so we use 2.2F as the max number of faces, edges, or vertices
    # plus a fudge factor for small meshes, where the relationships become less accurate.
    # If 2.2*length(faces) < typemax(UInt), convert to UInt
    if ceil(2.2*length(faces)) < typemax(UInt16)
        I = UInt16
        faces_16 = convert(Vector{Vector{UInt16}}, faces)
        face_sets_16 = convert(Dict{String, Set{UInt16}}, face_sets) 
        return UnstructuredMesh_2D{float_type, I}(name = name,
                                                  points = points,
                                                  faces = [ Tuple(f) for f in faces_16],
                                                  face_sets = face_sets_16
                                                 )
    elseif ceil(2.2*length(faces)) < typemax(UInt32)
        I = UInt32
        faces_32 = convert(Vector{Vector{UInt32}}, faces)
        face_sets_32 = convert(Dict{String, Set{UInt32}}, face_sets) 
        return UnstructuredMesh_2D{float_type, I}(name = name,
                                                  points = points,
                                                  faces = [ Tuple(f) for f in faces_32],
                                                  face_sets = face_sets_32
                                                 )
    else
        I = UInt64
        return UnstructuredMesh_2D{float_type, I}(name = name,
                                                  points = points,
                                                  faces = [ Tuple(f) for f in faces],
                                                  face_sets = face_sets
                                                 )
    end
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
        @error "$element_type is not in the abaqus to vtk type conversion dictionary"
    end
    type = abaqus_to_vtk_type[element_type]
    faces = Vector{UInt64}[]
    line_split = strip.(split(readline(file)), [','])
    line_position = position(file)
    while !occursin("*", line_split[1])
        vertex_IDs = parse.(UInt64, line_split[2:length(line_split)])
        if length(vertex_IDs) == 9
            push!(faces, vcat(type, vertex_IDs[1:8]))
        else
            push!(faces, vcat(type, vertex_IDs))
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
    return Set(parse.(UInt64, line_split))
end
