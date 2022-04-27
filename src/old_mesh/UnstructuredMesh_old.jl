abstract type UnstructuredMesh{Dim, Ord, T, U} end
const UnstructuredMesh2D = UnstructuredMesh{2}
const UnstructuredMesh3D = UnstructuredMesh{3}
const LinearUnstructuredMesh = UnstructuredMesh{Dim, 1} where {Dim}
const LinearUnstructuredMesh2D = UnstructuredMesh{2, 1}
const LinearUnstructuredMesh3D = UnstructuredMesh{3, 1}
const QuadraticUnstructuredMesh = UnstructuredMesh{Dim, 2} where {Dim}
const QuadraticUnstructuredMesh2D = UnstructuredMesh{2, 2}
const QuadraticUnstructuredMesh3D = UnstructuredMesh{3, 2}
Base.broadcastable(mesh::UnstructuredMesh) = Ref(mesh)

function Base.show(io::IO, mesh::UnstructuredMesh)
    mesh_type = typeof(mesh)
    println(io, mesh_type)
    println(io, "  ├─ Name      : $(mesh.name)")
    size_MB = Base.summarysize(mesh)/1E6
    if size_MB < 1
        size_KB = size_MB*1000
        println(io, "  ├─ Size (KB) : $size_KB")
    else
        println(io, "  ├─ Size (MB) : $size_MB")
    end
    println(io, "  ├─ Points    : $(length(mesh.points))")
    println(io, "  ├─ Faces     : $(length(mesh.faces))")
    println(io, "  └─ Face sets : $(length(keys(mesh.face_sets)))")
end

function submesh(name::String, mesh::UnstructuredMesh2D{Ord, T, U}) where {Ord, T, U}
    # Setup faces and get all vertex ids
    face_ids = mesh.face_sets[name]
    submesh_faces = [MVector(mesh.faces[id].data) for id ∈ face_ids]
    vertex_ids = U[]
    # This can be sped up substantially by keeping vertex_ids sorted and 
    # checking membership using binary search
    for face in submesh_faces
        for vid in face
            error("finish this")
            if !insorted(vid, vertex_ids)
                index = searchsortedfirst(vertex_ids, vid)
                insert!(vertex_ids, index, vid)
            end
            vindex = searchsortedfirst(vertex_ids, id)
            if id ∉ vertex_ids
                push!(vertex_ids, id)
            end
        end
    end
    # Need to remap vertex ids in faces to new ids
    sort!(vertex_ids)
    vertex_map = Dict{U, U}()
    for (i, v) in enumerate(vertex_ids)
        vertex_map[v] = i
    end
    points = Vector{Point2D{T}}(undef, length(vertex_ids))
    for (i, v) in enumerate(vertex_ids)
        points[i] = mesh.points[v]
    end
    # remap vertex ids in faces
    for face in submesh_faces
        for (i, v) in enumerate(face)
            face[i] = vertex_map[v]
        end
    end
    # At this point we have points, faces, & name.
    # Just need to get the face sets
    face_sets = Dict{String, BitSet}()
    for face_set_name in keys(mesh.face_sets)
        set_intersection = mesh.face_sets[face_set_name] ∩ face_ids
        if length(set_intersection) !== 0
            face_sets[face_set_name] = set_intersection
        end
    end
    # Need to remap face ids in face sets
    face_map = Dict{Int64, Int64}()
    for (i, f) in enumerate(face_ids)
        face_map[f] = i
    end
    for face_set_name in keys(face_sets)
        face_sets[face_set_name] = BitSet(map(x->face_map[x], 
                                              collect(face_sets[face_set_name])))
    end
    faces = [SVector{length(f), U}(f) for f in submesh_faces]
    return typeof(mesh)(name = name,
             points = points,
             faces = faces, 
             face_sets = face_sets
            )
end

#########################################################################################
function _create_2d_mesh_from_vector_faces(name::String, points::Vector{Point2D{T}},
                                           faces_vecs::Vector{Vector{UInt64}},
                                           face_sets::Dict{String, BitSet}
                                          ) where {T}
    # Determine face types
    face_lengths = Int64[]
    for face in faces_vecs
        l = length(face)
        if l ∉ face_lengths
            push!(face_lengths, l)
        end
    end
    sort!(face_lengths)
    if all(x->x < 6, face_lengths) # Linear mesh
        if face_lengths == [3]
            U = _select_mesh_UInt_type(max(length(points), length(faces_vecs)))
            faces = [ SVector{3, U}(f) for f in faces_vecs]
            return TriangleMesh{T, U}(name = name,
                                      points = points,
                                      faces = faces,
                                      face_sets = face_sets)
        elseif face_lengths == [4]
            U = _select_mesh_UInt_type(max(length(points), length(faces_vecs)))
            faces = [ SVector{4, U}(f) for f in faces_vecs]
            return QuadrilateralMesh{T, U}(name = name,
                                           points = points,
                                           faces = faces,
                                           face_sets = face_sets)
        elseif face_lengths == [3, 4]
            U = _select_mesh_UInt_type(max(length(points), length(faces_vecs)))
            faces = [ SVector{length(f), U}(f) for f in faces_vecs]
            return PolygonMesh{T, U}(name = name,
                                     points = points,
                                     faces = faces,
                                     face_sets = face_sets)
        end
    else # Quadratic Mesh
        if face_lengths == [6]
            U = _select_mesh_UInt_type(max(length(points), length(faces_vecs)))
            faces = [ SVector{6, U}(f) for f in faces_vecs]
            return QuadraticTriangleMesh{T, U}(name = name,
                                               points = points,
                                               faces = faces,
                                               face_sets = face_sets)
        elseif face_lengths == [8]
            U = _select_mesh_UInt_type(max(length(points), length(faces_vecs)))
            faces = [ SVector{8, U}(f) for f in faces_vecs]
            return QuadraticQuadrilateralMesh{T, U}(name = name,
                                                    points = points,
                                                    faces = faces,
                                                    face_sets = face_sets)
        elseif face_lengths == [6, 8]
            U = _select_mesh_UInt_type(max(length(points), length(faces_vecs)))
            faces = [ SVector{length(f), U}(f) for f in faces_vecs]
            return QuadraticPolygonMesh{T, U}(name = name,
                                              points = points,
                                              faces = faces,
                                              face_sets = face_sets)
        end
    end
    error("Could not determine mesh type")
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
