export submesh

is_MPACT_cell(x::String) = startswith(x, "Cell_")
is_MPACT_module(x::String) = startswith(x, "Module_")
is_MPACT_lattice(x::String) = startswith(x, "Lattice_")

function is_MPACT_partition(x::String)
    return is_MPACT_cell(x) || is_MPACT_module(x) || is_MPACT_lattice(x)
end

function _submesh_elsets(elsets::Dict{String, Set{I}},
                         name::String) where {I <: Integer}
    # Submesh elsets + remap
    # These intersection operations can be expernsive, so if the name by which
    # the mesh is being submeshed is part of the MPACT spatial hierarchy (i.e.
    # Lattice, Module, or Cell), then we want to use that information to
    # skip the appropriate intersection operations.
    # If name == "Lattice", then we don't need to do any intersections with
    # other lattices.
    # If name == "Module", then we don't need to do any intersections with
    # other lattices or modules.
    # If name == "Cell", then we don't need to do any intersections with
    # other lattices, modules, or cells.
    # It is assumed if you're submeshing one of these entities, you've already
    # submeshed the entities above it in the hierarchy.
    name_is_lattice = is_MPACT_lattice(name)
    name_is_module = is_MPACT_module(name)
    name_is_cell = is_MPACT_cell(name)
    sub_elsets = Dict{String, Set{I}}()
    face_ids = elsets[name]
    for set_name in keys(elsets)
        if name_is_cell
            if is_MPACT_partition(set_name)
                continue
            end
        elseif name_is_module 
            if is_MPACT_module(set_name) || is_MPACT_lattice(set_name)
                continue
            end
        elseif name_is_lattice 
            if is_MPACT_lattice(set_name)
                continue
            end
        end
        set_intersection = intersect(elsets[set_name], face_ids)
        if length(set_intersection) !== 0 && name !== set_name
            set_intersection_vec = collect(set_intersection)
            sub_elsets[set_name] = Set{I}(searchsortedfirst.(Ref(face_ids_vec), 
                                                                set_intersection_vec))
        end
    end
    return sub_elsets
end

function submesh(mesh::PolygonMesh{N, T, I},
                 elsets::Dict{String, Set{I}},
                 name::String) where {N, T <: AbstractFloat, I <: Integer}
    # Submesh faces + point ids
    face_ids = elsets[name]
    face_ids_vec = sort!(collect(face_ids))
    vert_ids_vec = I[]
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
    sub_elsets = _submesh_elsets(elsets, name)

    # Recompute vf_conn
    vf_offsets, vf_conn = polygon_mesh_vf_conn(N, nverts, fv_conn)

    return sub_elsets, PolygonMesh{N, T, I}(name, vertices, fv_conn, vf_offsets, vf_conn)
end

function submesh(mesh::QPolygonMesh{N, T, I},
                 elsets::Dict{String, Set{I}},
                 name::String) where {N, T <: AbstractFloat, I <: Integer}
    # Submesh faces + point ids
    face_ids = elsets[name]
    face_ids_vec = sort!(collect(face_ids))
    vert_ids_vec = I[]
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
    sub_elsets = _submesh_elsets(elsets, name)

    # Recompute vf_conn
    vf_offsets, vf_conn = polygon_mesh_vf_conn(N, nverts, fv_conn)

    return sub_elsets, QPolygonMesh{N, T, I}(name, vertices, fv_conn, vf_offsets, vf_conn)
end
