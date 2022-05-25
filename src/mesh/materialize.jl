export materialize, 
       materialize_face,
       materialize_cell,
       materialize_polytopes, 
       materialize_facets,
       materialize_cells, 
       materialize_faces, 
       materialize_edges

# Not type-stable
function materialize_face(i::Integer, mesh::VolumeMesh)
    vtk_type = mesh.types[i]
    Δ = mesh.offsets[i]
    points = mesh.points
    conn = mesh.connectivity
    if vtk_type == VTK_TRIANGLE
        vids = conn[Vec(ntuple(i->i+Δ-1, Val(3))...)]
        return Triangle(points[convert(SVector{3,Int64}, vids)])
    elseif vtk_type == VTK_QUAD
        vids = conn[Vec(ntuple(i->i+Δ-1, Val(4))...)]
        return Quadrilateral(points[convert(SVector{4,Int64}, vids)])
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        vids = conn[Vec(ntuple(i->i+Δ-1, Val(6))...)]
        return QuadraticTriangle(points[convert(SVector{6,Int64}, vids)])
    elseif vtk_type == VTK_QUADRATIC_QUAD
        vids = conn[Vec(ntuple(i->i+Δ-1, Val(8))...)]
        return QuadraticQuadrilateral(points[convert(SVector{8,Int64}, vids)])
    else
        error("Unsupported type.")
        return nothing
    end
end

# Not type-stable
function _materialize_face_connectivity(i::Integer, mesh::VolumeMesh)
    vtk_type = mesh.types[i]
    Δ = mesh.offsets[i]
    conn = mesh.connectivity
    if vtk_type == VTK_TRIANGLE
        vids = conn[Vec(ntuple(i->i+Δ-1, Val(3))...)]
        return Triangle(vids...)
    elseif vtk_type == VTK_QUAD
        vids = conn[Vec(ntuple(i->i+Δ-1, Val(4))...)]
        return Quadrilateral(vids...)
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        vids = conn[Vec(ntuple(i->i+Δ-1, Val(6))...)]
        return QuadraticTriangle(vids...)
    elseif vtk_type == VTK_QUADRATIC_QUAD
        vids = conn[Vec(ntuple(i->i+Δ-1, Val(8))...)]
        return QuadraticQuadrilateral(vids...)
    else
        error("Unsupported type.")
        return nothing
    end
end

function materialize_faces(mesh::VolumeMesh)
    return map(i->materialize_face(i, mesh), eachindex(mesh.types))
end

function materialize(p::Polytope{K,P,N,T}, vertices::Vector{Point{Dim,F}}
                    ) where {K,P,N,T,Dim,F}
    return Polytope{K,P,N,Point{Dim,F}}(
            Vec(ntuple(i->vertices[p.vertices[i]], Val(N)))
           ) 
end

function materialize_polytopes(mesh::PolytopeVertexMesh)
    return materialize.(mesh.polytopes, Ref(mesh.vertices))
end
#
## aliases
#function materialize_cells(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Cell}
#    return materialize_polytopes(mesh) 
#end
#function materialize_facets(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Cell}
#    return materialize_faces(mesh)
#end
#function materialize_ridges(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Cell}
#    return materialize_edges(mesh)
#end
#function materialize_faces(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Face}
#    return materialize_polytopes(mesh) 
#end
#function materialize_facets(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Face}
#    return materialize_edges(mesh)
#end
#
#function materialize_faces(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P}
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
#function materialize_edges(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:Polygon}
#    unique_edges = SVector{2,vertex_type(P)}[]
#    nedges = 0
#    for face ∈ mesh.polytopes
#        edge_vecs = edges(face)
#        for edge ∈ edge_vecs
#            if edge[1] < edge[2]
#                sorted_edge = edge.vertices
#            else
#                sorted_edge = SVector(edge[2], edge[1])
#            end
#            index = getsortedfirst(unique_edges, sorted_edge)
#            if nedges < index || unique_edges[index] !== sorted_edge
#                insert!(unique_edges, index, sorted_edge)
#                nedges += 1
#            end
#        end
#    end
#    materialized_edges = Vector{LineSegment{Point{Dim,T}}}(undef, nedges)
#    for i = 1:nedges
#        materialized_edges[i] = materialize(LineSegment(unique_edges[i]), mesh.vertices)
#    end
#    return materialized_edges 
#end 
#
## All edges are quadratic segments
#function materialize_edges(mesh::PolytopeVertexMesh{Dim,T,P}) where {Dim,T,P<:QuadraticPolygon}
#    unique_edges = SVector{3,vertex_type(P)}[]
#    nedges = 0
#    for face ∈ mesh.polytopes
#        edge_vecs = edges(face)
#        for edge ∈ edge_vecs
#            if edge[1] < edge[2]
#                sorted_edge = edge.vertices
#            else
#                sorted_edge = SVector(edge[2], edge[1], edge[3])
#            end
#            index = getsortedfirst(unique_edges, sorted_edge)
#            if nedges < index || unique_edges[index] !== sorted_edge
#                insert!(unique_edges, index, sorted_edge)
#                nedges += 1
#            end
#        end
#    end
#    materialized_edges = Vector{QuadraticSegment{Point{Dim,T}}}(undef, nedges)
#    for i = 1:nedges
#        materialized_edges[i] = materialize(QuadraticSegment(unique_edges[i]), mesh.vertices)
#    end
#    return materialized_edges 
#end 
