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
    faces_vecs = Vector{UInt64}[] 
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
                read_abaqus_nodes_2d!(file, points)
            elseif occursin("*ELEMENT", line)
                linesplit = split(line)
                element_type = String(strip(replace(linesplit[2], ("type=" => "")), ','))
                read_abaqus_elements!(file, faces_vecs, element_type)
            elseif occursin("*ELSET", line)
                linesplit = split(line)
                set_name = String(replace(linesplit[1], ("*ELSET,ELSET=" => "")))
                face_sets[set_name] = read_abaqus_elset(file)
            end
        end
    end
    close(file)
    # Determine face types
    face_lengths = Int64[]
    for face in faces_vecs  
        l = length(face)
        if l ∉ face_lengths
            push!(face_lengths, l)
        end
    end
    if all(x->x < 6, face_lengths) # Linear mesh  
        if face_lengths == [3]
            faces = [ SVector{3, UInt64}(f) for f in faces_vecs]
            edges = _create_linear_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U, face_sets_U = _convert_faces_edges_facesets(U, faces, edges, 
                                                                          face_sets)
            return TriangleMesh{2,floattype, U}(name = name,
                                          points = points,
                                          edges = edges_U,
                                          faces = faces_U,
                                          face_sets = face_sets_U)
        elseif face_lengths == [4]
            faces = [ SVector{4, UInt64}(f) for f in faces_vecs]
            edges = _create_linear_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U, face_sets_U = _convert_faces_edges_facesets(U, faces, edges, 
                                                                          face_sets)
            return QuadrilateralMesh{2,floattype, U}(name = name,
                                          points = points,
                                          edges = edges_U,
                                          faces = faces_U, 
                                          face_sets = face_sets_U)
        else
            faces = [ SVector{length(f), UInt64}(f) for f in faces_vecs]
            edges = _create_linear_edges_from_faces(faces)
            U = _select_mesh_UInt_type(length(edges))
            faces_U, edges_U, face_sets_U = _convert_faces_edges_facesets(U, faces, edges, 
                                                                          face_sets)
            return PolygonMesh{2,floattype, U}(name = name,
                                          points = points,
                                          edges = edges_U,
                                          faces = [ SVector{length(f), U}(f) for f in faces],
                                          face_sets = face_sets_U)
        end
    else # Quadratic Mesh
        if face_lengths == [6]
            return QuadraticTriangleMesh{2,floattype, U}(name = name,
                                          points = points,
                                          faces = [ SVector{6, U}(f) for f in faces],
                                          face_sets = face_sets_U)
        elseif face_lengths == [8]
            return QuadraticQuadrilateralMesh{2,floattype, U}(name = name,
                                          points = points,
                                          faces = [ SVector{8, U}(f) for f in faces],
                                          face_sets = face_sets_U)
        else
            return QuadraticPolygonMesh{2,floattype, U}(name = name,
                                          points = points,
                                          faces = [ SVector{length(f), U}(f) for f in faces],
                                          face_sets = face_sets_U)
        end
    end
end

function read_abaqus_nodes_2d!(file::IOStream, points::Vector{Point2D{T}}) where {T}
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
    new_points = Vector{Point2D{T}}(undef, npoints)
    line = readline(file)
    ipt = 0
    while !('*' == line[1])
        ipt += 1
        xy = parse.(T, strip.(view(split(line),2:3), [',']))
        new_points[ipt] = Point2D(xy[1], xy[2])
        file_position = position(file)
        line = readline(file)
    end
    seek(file, file_position)
    append!(points, new_points)
    return nothing
end

function read_abaqus_elements!(file::IOStream, faces::Vector{Vector{UInt64}}, 
                               element_type::String)
    if !(element_type ∈ valid_abaqus_types)  
        @error "$element_type is not in the valid abaqus types"
    end
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



function _create_linear_edges_from_faces(faces::Vector{<:SArray{S, U, 1} where {S<:Tuple}}
                                        ) where {U<:Unsigned}
    edge_vecs = linear_edges.(faces)
    num_edges = mapreduce(x->length(x), +, edge_vecs)
    edges_unfiltered = Vector{SVector{2, UInt64}}(undef, num_edges)
    iedge = 1
    for edge in edge_vecs
        for i in eachindex(edge)
            if edge[i][1] < edge[i][2]
                edges_unfiltered[iedge] = edge[i]
            else
                edges_unfiltered[iedge] = SVector(edge[i][2], edge[i][1])
            end
            iedge += 1
        end
    end
    return sort!(unique!(edges_unfiltered))
end

function _select_mesh_UInt_type(N::Int64)
    if N ≤ typemax(UInt16) 
        U = UInt16
    elseif N ≤ typemax(UInt32) 
        U = UInt32
    elseif N ≤ typemax(UInt64) 
        U = UInt64
    else 
        @error "How cow, that's a big mesh! Number of edges exceeds typemax(UInt64)"
        U = UInt64
    end
    return U
end

function _convert_faces_edges_facesets(::Type{U}, 
                                       faces::Vector{<:SArray{S, UInt64, 1} where {S<:Tuple}}, 
                                       edges::Vector{<:SVector{N, UInt64}}, 
                                       face_sets::Dict{String, Set{UInt64}}
                                      ) where {U, N}
    faces_U = [ convert(SVector{length(face), U}, face) for face in faces ]
    edges_U = [ convert(SVector{length(edge), U}, edge) for edge in edges ] 
    face_sets_U = Dict{String, Set{U}}()
    for key in keys(face_sets)
        face_sets_U[key] = convert(Set{U}, face_sets[key])
    end
    face_sets = nothing
    return faces_U, edges_U, face_sets_U 

end
