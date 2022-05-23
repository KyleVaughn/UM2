export submesh

function submesh(mesh::UnstructuredMesh{Dim,T,U}, name::String) where {Dim,T,U}
    # Submesh cells prior to point id remap 
    cell_ids = mesh.groups[name]
    ncells = length(cell_ids)
    submesh_cell_types = Vector{U}(undef, 2*ncells)
    cell_array_len = 1
    for (i, id) in enumerate(cell_ids)
        type   = mesh.cell_types[2id-1]
        offset = mesh.cell_types[2id  ]
        submesh_cell_types[2i-1] = type
        submesh_cell_types[2i  ] = cell_array_len
        npts = mesh.cell_array[offset]
        cell_array_len += npts + 1 
    end
    submesh_cell_array = Vector{U}(undef, cell_array_len-1)
    for (i, id) in enumerate(cell_ids)
        offset = mesh.cell_types[2id]
        npts = mesh.cell_array[offset]
        submesh_offset = submesh_cell_types[2i]
        submesh_cell_array[submesh_offset] = npts
        submesh_cell_array[submesh_offset+1:submesh_offset+npts] = 
            mesh.cell_array[offset+1:offset+npts]
    end

    # Submesh points 
    point_ids = BitSet()
    discard = 1
    for (i, vid) in enumerate(submesh_cell_array)
        if i == discard
            discard += vid + 1
            continue
        end
        push!(point_ids, vid)
    end
    npoints = length(point_ids)
    points = Vector{Point{Dim,T}}(undef, npoints)
    for (i, vid) in enumerate(point_ids)
        points[i] = mesh.points[vid]
    end

    # Remap point ids in cells
    point_ids_vec = collect(point_ids)
    discard = 1
    for (i, vid) in enumerate(submesh_cell_array)
        if i == discard
            discard += vid + 1
            continue
        end
        submesh_cell_array[i] = searchsortedfirst(point_ids_vec, vid)
    end

#    # Submesh groups prior to cell id remap 
#    groups = Dict{String,BitSet}()
#    for group_name in keys(mesh.groups)
#        set_intersection = mesh.groups[group_name] ∩ cell_ids
#        if length(set_intersection) !== 0 && name !== group_name
#            groups[group_name] = set_intersection
#        end
#    end
#
#    # Remap cell ids in groups
#    cell_ids_vec = collect(cell_ids)
#    for group_name in keys(groups)
#        groups[group_name] = BitSet(
#                                searchsortedfirst.(Ref(cell_ids_vec),
#                                                   groups[group_name] 
#                                )
#                             )
#    end
#
    return UnstructuredMesh{Dim,T,U}(points, submesh_cell_array, submesh_cell_types,
                                     name, Dict{String,BitSet}())
end

#function submesh(mesh::PolytopeVertexMesh, name::String)
#    # Submesh polytopes prior to vertex id remap 
#    polytope_ids = mesh.groups[name]
#    submesh_polytopes = [mesh.polytopes[pid] for pid in polytope_ids]
#
#    # Submesh points 
#    vertex_ids = BitSet()
#    for polytope in submesh_polytopes 
#        for vid in vertices(polytope)
#            push!(vertex_ids, vid)
#        end
#    end
#    npoints = length(vertex_ids)
#    point_type = typeof(vertices(mesh)[1])
#    points = Vector{point_type}(undef, npoints)
#    for (i, vid) in enumerate(vertex_ids)
#        points[i] = mesh.vertices[vid]
#    end
#
#    # Remap vertex ids in polytopes
#    vertex_ids_vec = collect(vertex_ids)
#    for i in eachindex(submesh_polytopes)
#        submesh_polytopes[i] = searchsortedfirst.(Ref(vertex_ids_vec), 
#                                                  vertices(submesh_polytopes[i])
#                               )
#    end
#
#    # Submesh groups prior to polytope id remap 
#    groups = Dict{String,BitSet}()
#    for group_name in keys(mesh.groups)
#        set_intersection = mesh.groups[group_name] ∩ polytope_ids
#        if length(set_intersection) !== 0 && name !== group_name
#            groups[group_name] = set_intersection
#        end
#    end
#
#    # Remap polytope ids in groups
#    polytope_ids_vec = collect(polytope_ids)
#    for group_name in keys(groups)
#        groups[group_name] = BitSet(
#                                searchsortedfirst.(Ref(polytope_ids_vec),
#                                                   groups[group_name] 
#                                )
#                             )
#    end
#
#    return typeof(mesh)(name, points, submesh_polytopes, groups)
#end
