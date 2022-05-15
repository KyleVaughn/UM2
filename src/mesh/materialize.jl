export materialize, materialize_polytopes, materialize_facets,
       materialize_cells, materialize_faces, materialize_edges

function materialize(poly::Polytope{K,P,N,T}, 
                     vertices::Vector{Point{Dim,F}}) where {K,P,N,T,Dim,F}
    return Polytope{K,P,N,Point{Dim,F}}(Vec(
            ntuple(i->vertices[poly.vertices[i]], Val(N))
    )) 
end

function materialize_polytopes(mesh::PolytopeVertexMesh)
    return materialize.(mesh.polytopes, Ref(mesh.vertices))
end

# aliases
function materialize_cells(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Cell}
    return materialize_polytopes(mesh) 
end
function materialize_facets(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Cell}
    return materialize_faces(mesh)
end
function materialize_ridges(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Cell}
    return materialize_edges(mesh)
end
function materialize_faces(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Face}
    return materialize_polytopes(mesh) 
end
function materialize_facets(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Face}
    return materialize_edges(mesh)
end

function materialize_faces(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P}
    return materialize.(
                unique!(
                    x->sort(x.vertices),
                    sort!(
                        reduce(vcat, 
                            faces.(mesh.polytopes)
                        )
                        ,by=x->sort(x.vertices), 
                    )
                ),
                Ref(mesh.vertices)
            )   
end

# TODO:
# Fast faces methods for Polyhedrons/quadratic polyhedrons where all elements are the same type

# All edges are line segments
function materialize_edges(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Polygon}
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
function materialize_edges(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:QuadraticPolygon}
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
