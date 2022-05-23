export submesh

function submesh(mesh::UnstructuredMesh{Dim,T,U}, name::String) where {Dim,T,U}
    # Submesh cells prior to point id remap 
    cell_ids = mesh.groups[name]
    submesh_cell_array = [mesh.cells[pid] for pid in cell_ids]

    # Submesh points 
    point_ids = BitSet()
    for cell in submesh_cells 
        for vid in vertices(cell)
            push!(point_ids, vid)
        end
    end
    npoints = length(point_ids)
    point_type = typeof(vertices(mesh)[1])
    points = Vector{point_type}(undef, npoints)
    for (i, vid) in enumerate(point_ids)
        points[i] = mesh.vertices[vid]
    end

    # Remap point ids in cells
    point_ids_vec = collect(point_ids)
    for i in eachindex(submesh_cells)
        submesh_cells[i] = searchsortedfirst.(Ref(point_ids_vec), 
                                                  vertices(submesh_cells[i])
                               )
    end

    # Submesh groups prior to cell id remap 
    groups = Dict{String,BitSet}()
    for group_name in keys(mesh.groups)
        set_intersection = mesh.groups[group_name] ∩ cell_ids
        if length(set_intersection) !== 0 && name !== group_name
            groups[group_name] = set_intersection
        end
    end

    # Remap cell ids in groups
    cell_ids_vec = collect(cell_ids)
    for group_name in keys(groups)
        groups[group_name] = BitSet(
                                searchsortedfirst.(Ref(cell_ids_vec),
                                                   groups[group_name] 
                                )
                             )
    end

    return typeof(mesh)(name, points, submesh_cells, groups)
end

function submesh(mesh::PolytopeVertexMesh, name::String)
    # Submesh polytopes prior to vertex id remap 
    polytope_ids = mesh.groups[name]
    submesh_polytopes = [mesh.polytopes[pid] for pid in polytope_ids]

    # Submesh points 
    vertex_ids = BitSet()
    for polytope in submesh_polytopes 
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
    for i in eachindex(submesh_polytopes)
        submesh_polytopes[i] = searchsortedfirst.(Ref(vertex_ids_vec), 
                                                  vertices(submesh_polytopes[i])
                               )
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

    return typeof(mesh)(name, points, submesh_polytopes, groups)
end
