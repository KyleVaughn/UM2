function edge_connectivity(mesh::VolumeMesh{2, T, U}) where {T, U}
    if islinear(mesh)
        return _volume_mesh_linear_edges(mesh)
    else
        return _volume_mesh_quadratic_edges(mesh)
    end
end

function _volume_mesh_linear_edges(mesh::VolumeMesh{2, T, U}) where {T, U}
    unique_edges = Vec{2,U}[]
    nedges = 0
    for face ∈ mesh.offsets
        edge_vecs = edges(face)
        for edge ∈ edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge.vertices
            else
                sorted_edge = Vec(edge[2], edge[1])
            end
            index = getsortedfirst(unique_edges, sorted_edge)
            if nedges < index || unique_edges[index] !== sorted_edge
                insert!(unique_edges, index, sorted_edge)
                nedges += 1
            end
        end
    end
    materialized_edges = Vector{LineSegment{Point{D,T}}}(undef, nedges)
    for i = 1:nedges
        materialized_edges[i] = materialize(LineSegment(unique_edges[i]), mesh.vertices)
    end
    return materialized_edges 
end 
