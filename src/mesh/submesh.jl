export submesh

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
        if length(set_intersection) !== 0
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
