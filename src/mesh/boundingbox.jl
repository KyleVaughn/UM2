function boundingbox(mesh::AbstractMesh)
    if islinear(mesh)
        return boundingbox(points(mesh))
    else
        quadratic_edges = materialize_edges(mesh)
        return mapreduce(boundingbox, ∪, quadratic_edges)
    end
end
