function Base.intersect(l::LineSegment{Point{2, T}},
                        mesh::PolytopeVertexMesh{2, T}) where {T}
    return Base.intersect(l, mesh.polytopes, mesh.vertices) 
end
