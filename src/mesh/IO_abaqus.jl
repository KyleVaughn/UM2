# IO routines for the Abaqus .inp file format
const valid_abaqus_types = (
    # 2D
    # triangle
    "CPS3", "SFRU3",
    # triangle6
    "CPS6",
    # quadrilateral
    "CPS4", 
    # quad8
    "CPS8",
    "M3D9"
   )

function read_abaqus2d(filepath::String, floattype::Type{T}=Float64) where {T<:AbstractFloat}
    @info "Reading "*filepath
    # NOTE: There is a critical assumption here that elements and nodes are listed 1 to N,
    # not 8, 10, 9 or anything funky/out of order.
    file = open(filepath, "r")
    name = "default_name"
    faces = [] 
    face_sets = Dict{String, Set{UInt64}}()
    points = Point2D{floattype}[]
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
                points = read_abaqus_nodes_2d(file, floattype)
            elseif occursin("*ELEMENT", line)
                linesplit = split(line)
                element_type = String(strip(replace(linesplit[2], ("type=" => "")), ','))
                append!(faces, read_abaqus_elements(file, element_type))
            elseif occursin("*ELSET", line)
                linesplit = split(line)
                set_name = String(replace(linesplit[1], ("*ELSET,ELSET=" => "")))
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
    UInt16_max = 65535
    UInt32_max = 4294967295
    nfaces = length(faces)
    if 2.2*nfaces + 1000 < UInt16_max
        U = UInt16
    elseif 2.2*nfaces + 1000 < UInt32_max 
        U = UInt32
    else
        U = UInt64
    end
    face_sets_U = Dict{String, Set{U}}()
    for key in keys(face_sets)
        face_sets_U[key] = convert(Set{U}, face_sets[key])
    end
    face_lengths = Int64[]
    for face in faces
        l = length(face)
        if l ∉ face_lengths
            push!(face_lengths, l)
        end
    end
    if face_lengths == [3]
        return TriangleMesh{2,floattype, U}(name = name,
                                           points = points,
                                           faces = [ SVector{3, U}(f) for f in faces],
                                           face_sets = face_sets_U)
    elseif face_lengths == [4]
        return QuadrilateralMesh{2,floattype, U}(name = name,
                                           points = points,
                                           faces = [ SVector{4, U}(f) for f in faces],
                                           face_sets = face_sets_U)
    
    else
        return PolygonMesh{2,floattype, U}(name = name,
                                           points = points,
                                           faces = [ SVector{length(f), U}(f) for f in faces],
                                           face_sets = face_sets_U)
    end
end

function read_abaqus_nodes_2d(file::IOStream, floattype::Type{T}) where {T<:AbstractFloat} 
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
    points = Vector{Point2D{floattype}}(undef, npoints)
    line = readline(file)
    ipt = 0
    while !('*' == line[1])
        ipt += 1
        xy = parse.(floattype, strip.(view(split(line),2:3), [',']))
        points[ipt] = Point2D(xy[1], xy[2])
        file_position = position(file)
        line = readline(file)
    end
    seek(file, file_position)
    return points
end

function read_abaqus_elements(file::IOStream, element_type::String)
    if !(element_type ∈ valid_abaqus_types)  
        @error "$element_type is not in the valid abaqus types"
    end
    faces = []
    line = readline(file)
    file_position = position(file)
    while !('*' == line[1] || eof(file))
        linesplit = split(line)
        vertexIDs = parse.(UInt64, strip.(view(linesplit, 2:length(linesplit)), [',']))
        if length(vertexIDs) == 9
            push!(faces, vertexIDs[1:8])
        else
            push!(faces, vertexIDs)
        end
        file_position = position(file)
        line = readline(file)
    end
    seek(file, file_position)
    return faces
end

function read_abaqus_elset(file::IOStream)
    linesplit = strip.(split(readuntil(file, "*")), [','])
    seek(file, position(file)-1)
    return Set(parse.(UInt64, linesplit))
end
