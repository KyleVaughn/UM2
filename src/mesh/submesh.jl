export submesh

function submesh(mesh::VolumeMesh{Dim,T,U}, name::String) where {Dim,T,U}
    # Submesh elements prior to point id remap 
    element_ids = mesh.groups[name]
    nelements = length(element_ids)
    types = Vector{U}(undef, nelements)
    offsets = Vector{U}(undef, nelements)
    materials = Vector{UInt8}(undef, nelements)
    connectivity_len = 1
    for (i, id) in enumerate(element_ids)
        types[i] = mesh.types[id]
        offsets[i] = connectivity_len
        materials[i] = mesh.materials[id]
        connectivity_len += points_in_vtk_type(types[i])
    end
    connectivity = Vector{U}(undef, connectivity_len - 1)
    for (i, id) in enumerate(element_ids)
        npts = points_in_vtk_type(types[i])
        sm_offset = offsets[i]
        m_offset = mesh.offsets[id]
        connectivity[sm_offset:sm_offset+npts-1] = mesh.connectivity[m_offset:m_offset+npts-1]
    end

    # Submesh points 
    point_ids = BitSet()
    for vid in connectivity
        push!(point_ids, vid)
    end
    npoints = length(point_ids)
    points = Vector{Point{Dim,T}}(undef, npoints)
    for (i, vid) in enumerate(point_ids)
        points[i] = mesh.points[vid]
    end

    # Remap point ids in connectivity 
    point_ids_vec = collect(point_ids)
    for (i, vid) in enumerate(connectivity)
        connectivity[i] = searchsortedfirst(point_ids_vec, vid)
    end

    # Submesh groups prior to element id remap 
    groups = Dict{String,BitSet}()
    for group_name in keys(mesh.groups)
        set_intersection = mesh.groups[group_name] ∩ element_ids
        if length(set_intersection) !== 0 && name !== group_name
            groups[group_name] = set_intersection
        end
    end

    # Remap element ids in groups
    element_ids_vec = collect(element_ids)
    for group_name in keys(groups)
        groups[group_name] = BitSet(
                                searchsortedfirst.(Ref(element_ids_vec),
                                                   groups[group_name] 
                                )
                             )
    end
    return VolumeMesh{Dim,T,U}(points, offsets, connectivity, types, 
                               materials, mesh.material_names, name, groups)
end

function submesh(mesh::PolytopeVertexMesh, name::String)
    # Submesh polytopes prior to vertex id remap 
    polytope_ids = mesh.groups[name]
    polytopes = [mesh.polytopes[pid] for pid in polytope_ids]

    # Submesh points 
    vertex_ids = BitSet()
    for polytope in polytopes 
        for vid in vertices(polytope)
            push!(vertex_ids, vid)
        end
    end
    npoints = length(vertex_ids)
    point_type = typeof(vertices(mesh)[1])
    points = Vector{point_type}(undef, npoints)
    for (i, vid) in enumerate(vertex_ids)
        points[i] = mesh.vertices[vid]
    end

    # Remap vertex ids in polytopes
    vertex_ids_vec = collect(vertex_ids)
    for i in eachindex(polytopes)
        polytopes[i] = searchsortedfirst.(Ref(vertex_ids_vec), vertices(polytopes[i]))
    end

    # Submesh groups prior to polytope id remap 
    groups = Dict{String,BitSet}()
    for group_name in keys(mesh.groups)
        set_intersection = mesh.groups[group_name] ∩ polytope_ids
        if length(set_intersection) !== 0 && name !== group_name
            groups[group_name] = set_intersection
        end
    end

    # Remap polytope ids in groups
    polytope_ids_vec = collect(polytope_ids)
    for group_name in keys(groups)
        groups[group_name] = BitSet(
                                searchsortedfirst.(Ref(polytope_ids_vec),
                                                   groups[group_name] 
                                )
                             )
    end

    materials = Vector{UInt8}(undef, length(polytope_ids_vec))
    for (i, id) in enumerate(polytope_ids_vec)
        materials[i] = mesh.materials[id]
    end
    return typeof(mesh)(points, polytopes, materials, 
                        mesh.material_names, name, groups)
end
