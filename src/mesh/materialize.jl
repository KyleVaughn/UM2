export materialize, materialize_polytopes, materialize_facets,
       materialize_faces, materialize_edges

@generated function materialize(poly::Polytope{K,P,N,T}, 
                                vertices::Vector{Point{Dim,F}}) where {K,P,N,T,Dim,F}
    exprs = [:(vertices[poly.vertices[$i]]) for i ∈ 1:N]
    return quote
        return Polytope{$K,$P,$N,Point{$Dim,$F}}(Vec($(exprs...))) 
    end
end

#materialize_faces(

function materialize_polytopes(mesh::PolytopeVertexMesh)
    return materialize.(mesh.polytopes, Ref(mesh.vertices))
end

function materialize_facets(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P}
    if P <: Edge
        return mesh.vertices
    end
    return materialize.(
                unique!(
                    x->sort(x.vertices),
                    sort!(
                        reduce(vcat, 
                            facets.(mesh.polytopes)
                        )
                        ,by=x->sort(x.vertices), 
                    )
                ),
                Ref(mesh.vertices)
            )   
end

# All edges are line segments
function materialize_facets(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Polygon}
    unique_edges = SVector{2,typeof(mesh.polytopes[1].vertices[1])}[]
    nedges = 0
    for face ∈ mesh.polytopes
        edge_vecs = edges(face)
        for edge ∈ edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge.vertices
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
    materialized_edges = Vector{LineSegment{Point{Dim,T}}}(undef, nedges)
    for i = 1:nedges
        materialized_edges[i] = materialize(LineSegment(unique_edges[i]), mesh.vertices)
    end
    return materialized_edges 
end 

# All edges are quadratic segments
function materialize_facets(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:QuadraticPolygon}
    unique_edges = SVector{3,typeof(mesh.polytopes[1].vertices[1])}[]
    nedges = 0
    for face ∈ mesh.polytopes
        edge_vecs = edges(face)
        for edge ∈ edge_vecs
            if edge[1] < edge[2]
                sorted_edge = edge.vertices
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
    materialized_edges = Vector{QuadraticSegment{Point{Dim,T}}}(undef, nedges)
    for i = 1:nedges
        materialized_edges[i] = materialize(QuadraticSegment(unique_edges[i]), mesh.vertices)
    end
    return materialized_edges 
end 
