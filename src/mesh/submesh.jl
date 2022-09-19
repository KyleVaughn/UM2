export submesh

function submesh(name::String, 
                 elsets::Dict{String, Set{Int32}},
                 mesh::PolygonMesh{N, T, I}) where {N, T, I}
    # Submesh faces + point ids
    face_ids = elsets[name]
    face_ids_vec = sort!(collect(face_ids))
    vert_ids_vec = Int32[]
    nverts = 0
    nfaces = length(face_ids_vec)
    fv_conn = Vector{I}(undef, N * nfaces)
    for (i, fid) in enumerate(face_ids_vec)
        fv_conn[(N * (i - 1) + 1):(N * i)] .= 
            mesh.fv_conn[(N * (fid - 1) + 1):(N * fid)]
        for j in (N * (fid - 1) + 1):(N * fid)
            n = searchsortedfirst(vert_ids_vec, mesh.fv_conn[j])
            if nverts < n || vert_ids_vec[n] !== mesh.fv_conn[j]
                insert!(vert_ids_vec, n, mesh.fv_conn[j])
                nverts += 1
            end
        end
    end

    # Submesh vertices 
    vertices = Vector{Point2{T}}(undef, nverts)
    for (i, vid) in enumerate(vert_ids_vec)
        vertices[i] = mesh.vertices[vid]
    end

    # Remap point ids in fv_conn
    for (i, vid) in enumerate(fv_conn)
        fv_conn[i] = searchsortedfirst(vert_ids_vec, vid)
    end

    # Submesh elsets + remap
    sub_elsets = Dict{String, Set{Int32}}()
    for set_name in keys(elsets)
        set_intersection = intersect(elsets[set_name], face_ids)
        if length(set_intersection) !== 0 && name !== set_name
            set_intersection_vec = collect(set_intersection)
            sub_elsets[set_name] = Set{Int32}(searchsortedfirst.(Ref(face_ids_vec), 
                                                                set_intersection_vec))
        end
    end

    # Recompute vf_conn
    vf_offsets, vf_conn = polygon_mesh_vf_conn(N, nverts, fv_conn)

    return sub_elsets, PolygonMesh{N, T, I}(name, vertices, fv_conn, vf_offsets, vf_conn)
end

function submesh(name::String, 
                 elsets::Dict{String, Set{Int32}},
                 mesh::QPolygonMesh{N, T, I}) where {N, T, I}
    # Submesh faces + point ids
    face_ids = elsets[name]
    face_ids_vec = sort!(collect(face_ids))
    vert_ids_vec = Int32[]
    nverts = 0
    nfaces = length(face_ids_vec)
    fv_conn = Vector{I}(undef, N * nfaces)
    for (i, fid) in enumerate(face_ids_vec)
        fv_conn[(N * (i - 1) + 1):(N * i)] .= 
            mesh.fv_conn[(N * (fid - 1) + 1):(N * fid)]
        for j in (N * (fid - 1) + 1):(N * fid)
            n = searchsortedfirst(vert_ids_vec, mesh.fv_conn[j])
            if nverts < n || vert_ids_vec[n] !== mesh.fv_conn[j]
                insert!(vert_ids_vec, n, mesh.fv_conn[j])
                nverts += 1
            end
        end
    end

    # Submesh vertices 
    vertices = Vector{Point2{T}}(undef, nverts)
    for (i, vid) in enumerate(vert_ids_vec)
        vertices[i] = mesh.vertices[vid]
    end

    # Remap point ids in fv_conn
    for (i, vid) in enumerate(fv_conn)
        fv_conn[i] = searchsortedfirst(vert_ids_vec, vid)
    end

    # Submesh elsets + remap
    sub_elsets = Dict{String, Set{Int32}}()
    for set_name in keys(elsets)
        set_intersection = intersect(elsets[set_name], face_ids)
        if length(set_intersection) !== 0 && name !== set_name
            set_intersection_vec = collect(set_intersection)
            sub_elsets[set_name] = Set{Int32}(searchsortedfirst.(Ref(face_ids_vec), 
                                                                set_intersection_vec))
        end
    end

    # Recompute vf_conn
    vf_offsets, vf_conn = polygon_mesh_vf_conn(N, nverts, fv_conn)

    return sub_elsets, QPolygonMesh{N, T, I}(name, vertices, fv_conn, vf_offsets, vf_conn)
end
