function submesh(name::String, mesh::UnstructuredMesh2D{Ord, T, U}) where {Ord,T,U}
    # Setup faces and get all vertex ids
    face_ids = mesh.face_sets[name]
    submesh_faces = [MVector(mesh.faces[id].data) for id ∈ face_ids]
    vertex_ids = U[]
    nverts = 0
    for face in submesh_faces
        for vid in face
            index = searchsortedfirst(vertex_ids, vid)
            # If the vid ∉ vertex_ids, insert it
            if nverts < index || vertex_ids[index] !== vid
                insert!(vertex_ids, index, vid)
                nverts += 1
            end
        end
    end
    points = mesh.points[vertex_ids] 
    # remap vertex ids in faces
    for face in submesh_faces
        for (i, v) in enumerate(face)
            face[i] = searchsortedfirst(vertex_ids, v)
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
