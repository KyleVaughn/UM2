# Return the point IDs representing the edges of a polygon.
@generated function polygon_edges(face::SVector{N, U}) where {N, U <:Unsigned}
    edges_string = "SVector{N, SVector{2, U}}("
    for i ∈ 1:N
        id₁ = (i - 1) % N + 1
        id₂ = i % N + 1
        edges_string *= "SVector{2, U}(face[$id₁], face[$id₂]), "
    end
    edges_string *= ")"
    return Meta.parse(edges_string)
end

@generated function quadratic_polygon_edges(face::SVector{N, U}) where {N, U <:Unsigned}
    M = N ÷ 2
    edges_string = "SVector{$M, SVector{3, U}}("
    for i ∈ 1:M
        id₁ = (i - 1) % M + 1
        id₂ = i % M + 1
        id₃ = i + M
        edges_string *= "SVector{3, U}(face[$id₁], face[$id₂], face[$id₃]), "
    end
    edges_string *= ")"
    return Meta.parse(edges_string)
end


function edges(mesh::PolygonMesh{T, U}) where {T, U}
    unique_edges = SVector{2, U}[]
    nedges = 0 
    for face ∈ mesh.faces
        edge_vecs = polygon_edges(face)
        for edge ∈ edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge
            else
                sorted_edge = SVector(edge[2], edge[1])
            end
            index = searchsortedfirst(unique_edges, sorted_edge)
            if nedges < index || unique_edges[index] !== sorted_edge
                insert!(unique_edges, index, sorted_edge)
                nedges += 1
            end 
        end
    end 
    return unique_edges
end

function edges(mesh::QuadraticPolygonMesh{T, U}) where {T, U}
    unique_edges = SVector{3, U}[]
    nedges = 0 
    for face ∈ mesh.faces
        edge_vecs = quadratic_polygon_edges(face)
        for edge ∈ edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge
            else
                sorted_edge = SVector(edge[2], edge[1], edge[3])
            end
            index = searchsortedfirst(unique_edges, sorted_edge)
            if nedges < index || unique_edges[index] !== sorted_edge
                insert!(unique_edges, index, sorted_edge)
                nedges += 1
            end 
        end
    end 
    return unique_edges
end
