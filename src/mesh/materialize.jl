export materialize,
       materialize_face,
       materialize_cell,
       materialize_polytopes,
       materialize_facets,
       materialize_cells,
       materialize_faces,
       materialize_edges

# Not type-stable
function materialize_face(i::Integer, mesh::VolumeMesh{2})
    offset = mesh.offsets[i]
    Δ = offset_diff(i, mesh)
    points = mesh.points
    conn = mesh.connectivity
    if Δ == 3
        vids = Vec{3, Int64}(conn[Vec(ntuple(i -> i + offset - 1, Val(3))...)])
        return Triangle(points[vids])
    elseif Δ == 4
        vids = Vec{4, Int64}(conn[Vec(ntuple(i -> i + offset - 1, Val(4))...)])
        return Quadrilateral(points[vids])
    elseif Δ == 6
        vids = Vec{6, Int64}(conn[Vec(ntuple(i -> i + offset - 1, Val(6))...)])
        return QuadraticTriangle(points[vids])
    elseif Δ == 8
        vids = Vec{8, Int64}(conn[Vec(ntuple(i -> i + offset - 1, Val(8))...)])
        return QuadraticQuadrilateral(points[vids])
    else
        error("Unsupported type.")
        return nothing
    end
end

# Not type-stable
function _materialize_face_connectivity(i::Integer, mesh::VolumeMesh{2})
    offset = mesh.offsets[i]
    Δ = offset_diff(i, mesh)
    conn = mesh.connectivity
    if Δ == 3
        vids = conn[Vec(ntuple(i -> i + offset - 1, Val(3))...)]
        return Triangle(vids)
    elseif Δ == 4
        vids = conn[Vec(ntuple(i -> i + offset - 1, Val(4))...)]
        return Quadrilateral(vids)
    elseif Δ == 6
        vids = conn[Vec(ntuple(i -> i + offset - 1, Val(6))...)]
        return QuadraticTriangle(vids)
    elseif Δ == 8
        vids = conn[Vec(ntuple(i -> i + offset - 1, Val(8))...)]
        return QuadraticQuadrilateral(vids)
    else
        error("Unsupported type.")
        return nothing
    end
end

function materialize_faces(mesh::VolumeMesh)
    return map(i -> materialize_face(i, mesh), 1:nelements(mesh))
end

function materialize(p::Polytope{K, P, N, T},
                     vertices::Vector{Point{D, F}}) where {K, P, N, T, D, F}
    return Polytope{K, P, N, Point{D, F}}(Vec(ntuple(i -> vertices[p.vertices[i]],
                                                     Val(N))))
end

function materialize_polytopes(mesh::PolytopeVertexMesh)
    return materialize.(mesh.polytopes, Ref(mesh.vertices))
end
#
## aliases
#function materialize_cells(mesh::PolytopeVertexMesh{D,T,P}) where {D,T,P<:Cell}
#    return materialize_polytopes(mesh) 
#end
#function materialize_facets(mesh::PolytopeVertexMesh{D,T,P}) where {D,T,P<:Cell}
#    return materialize_faces(mesh)
#end
#function materialize_ridges(mesh::PolytopeVertexMesh{D,T,P}) where {D,T,P<:Cell}
#    return materialize_edges(mesh)
#end
#function materialize_faces(mesh::PolytopeVertexMesh{D,T,P}) where {D,T,P<:Face}
#    return materialize_polytopes(mesh) 
#end
#function materialize_facets(mesh::PolytopeVertexMesh{D,T,P}) where {D,T,P<:Face}
#    return materialize_edges(mesh)
#end
#
#function materialize_faces(mesh::PolytopeVertexMesh{D,T,P}) where {D,T,P}
#    # Get the faces for each polytope, then reduce into a single vector.
#    # Sort the vector by each face's vertices, then get the unique faces.
#    # Materialize the faces.
#    return materialize.(
#                unique!(
#                    x->sort(x.vertices),
#                    sort!(
#                        reduce(vcat, 
#                            faces.(mesh.polytopes)
#                        )
#                        ,by=x->sort(x.vertices), 
#                    )
#                ),
#                Ref(mesh.vertices)
#            )   
#end
#
## TODO:
## Fast faces methods for Polyhedrons/quadratic polyhedrons where all elements are the same type
#
## All edges are line segments
#function materialize_edges(mesh::PolytopeVertexMesh{D,T,P}) where {D,T,P<:Polygon}
#    unique_edges = Vec{2,vertex_type(P)}[]
#    nedges = 0
#    for face ∈ mesh.polytopes
#        edge_vecs = edges(face)
#        for edge ∈ edge_vecs
#            if edge[1] < edge[2]
#                sorted_edge = edge.vertices
#            else
#                sorted_edge = Vec(edge[2], edge[1])
#            end
#            index = getsortedfirst(unique_edges, sorted_edge)
#            if nedges < index || unique_edges[index] !== sorted_edge
#                insert!(unique_edges, index, sorted_edge)
#                nedges += 1
#            end
#        end
#    end
#    materialized_edges = Vector{LineSegment{Point{D,T}}}(undef, nedges)
#    for i = 1:nedges
#        materialized_edges[i] = materialize(LineSegment(unique_edges[i]), mesh.vertices)
#    end
#    return materialized_edges 
#end 
#
## All edges are quadratic segments
#function materialize_edges(mesh::PolytopeVertexMesh{D,T,P}) where {D,T,P<:QuadraticPolygon}
#    unique_edges = Vec{3,vertex_type(P)}[]
#    nedges = 0
#    for face ∈ mesh.polytopes
#        edge_vecs = edges(face)
#        for edge ∈ edge_vecs
#            if edge[1] < edge[2]
#                sorted_edge = edge.vertices
#            else
#                sorted_edge = Vec(edge[2], edge[1], edge[3])
#            end
#            index = getsortedfirst(unique_edges, sorted_edge)
#            if nedges < index || unique_edges[index] !== sorted_edge
#                insert!(unique_edges, index, sorted_edge)
#                nedges += 1
#            end
#        end
#    end
#    materialized_edges = Vector{QuadraticSegment{Point{D,T}}}(undef, nedges)
#    for i = 1:nedges
#        materialized_edges[i] = materialize(QuadraticSegment(unique_edges[i]), mesh.vertices)
#    end
#    return materialized_edges 
#end 
