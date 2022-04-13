function edge_statistics(mesh::PolygonMesh; plot::Bool = false)
    mesh_edges = materialize_edges(mesh)
    lengths = measure.(materialize_edges(mesh))
    mean_length = mean(lengths)
    median_length = median(lengths)
    min_length = minimum(lengths)
    max_length = maximum(lengths)
    std_length = std(lengths)

end
# Straightness
