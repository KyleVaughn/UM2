# IO routines for the Abaqus .inp file format
const abaqus_to_vtk_type = Dict{String, UInt64}(
    # 2D
    # triangle
    "CPS3"  => 5,
    "SFRU3" => 5,
    # triangle6
    "CPS6" => 22,
    # quadrilateral
    "CPS4" => 9,
    # quad8
    "CPS8" => 23,
    "M3D9" => 23
   )

function read_abaqus_2d(filepath::String)
    @info "Reading $filepath"
    # NOFE: Fhere is a crucial assumption here that elements and nodes are listed 1 to N,
    # not 8, 10, 9 or anything funky/out of order.
    name = "DefaultMeshName"
    file = open(filepath, "r")
    faces = Vector{UInt32}[]
    face_sets = Dict{String, Set{UInt32}}()
    points = Point_2D[]
    while !eof(file)
        line_split = split(readline(file))
        if length(line_split) > 0
            if occursin("**", line_split[1]) # Comment
                continue
            elseif occursin("*Heading", line_split[1])
                name = String(strip(readline(file)))
                if occursin(".inp", name)
                    name = name[1:length(name)-4]
                end
            elseif occursin("*NODE", line_split[1])
                points = read_abaqus_nodes_2d(file)
            elseif occursin("*ELEMENT", line_split[1])
                element_type = String(strip(replace(line_split[2], ("type=" => "")), ','))
                append!(faces, read_abaqus_elements(file, element_type))
            elseif occursin("*ELSET", line_split[1])
                set_name = String(replace(line_split[1], ("*ELSET,ELSET=" => "")))
                face_sets[set_name] = read_abaqus_elset(file)
            end
        end
    end
    close(file)
    # The relationship between faces, edges, and vertices is
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
    # We see that V ≤ 2F ≤ E, so we use 2.2F as the max number of faces, edges, or vertices
    # plus a fudge factor for small meshes, where the relationships become less accurate.
    UInt32_max = 4294967295
    nfaces = length(faces)
    if UInt32_max < 2.2*nfaces
        @error "Mesh may cause UInt32 overflow. Some poor dev now has to add UInt64 support"
    end
    faces_total_length = mapreduce(x->length(x), +, faces)
    if faces_total_length % 3 === 0
        return TriangleMesh_2D(name = name,
                               points = points,
                               faces = [ SVector{3, UInt32}(f) for f in faces],
                               face_sets = face_sets
                              )
    elseif faces_total_length % 4 === 0
        return QuadrilateralMesh_2D(name = name,
                                    points = points,
                                    faces = [ SVector{4, UInt32}(f) for f in faces],
                                    face_sets = face_sets
                                   )
    else
        @error "Could not identify mesh type"
        return nothing
    end
end

function read_abaqus_nodes_2d(file::IOStream)
    points = Point_2D[]
    line_split = strip.(split(readline(file)), [','])
    line_position = position(file)
    while !occursin("*", line_split[1])
        xyz = parse.(Float64, line_split[2:4])
        push!(points, Point_2D(xyz[1], xyz[2]))
        line_position = position(file)
        line_split = strip.(split(readline(file)), [','])
    end
    seek(file, line_position)
    return points
end

function read_abaqus_elements(file::IOStream, element_type::String)
    if !(element_type ∈  keys(abaqus_to_vtk_type))
        @error "$element_type is not in the abaqus to vtk type conversion dictionary"
    end
    faces = Vector{UInt32}[]
    line_split = strip.(split(readline(file)), [','])
    line_position = position(file)
    while !occursin("*", line_split[1])
        vertex_IDs = parse.(UInt32, line_split[2:length(line_split)])
        if length(vertex_IDs) == 9
            push!(faces, vertex_IDs[1:8])
        else
            push!(faces, vertex_IDs)
        end
        line_position = position(file)
        line_split = strip.(split(readline(file)), [','])
    end
    seek(file, line_position)
    return faces
end

function read_abaqus_elset(file::IOStream)
    line_split = strip.(split(readuntil(file, "*")), [','])
    seek(file, position(file)-1)
    return Set(parse.(UInt32, line_split))
end
