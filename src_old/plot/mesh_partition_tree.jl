function convert_arguments(LS::Type{<:LineSegments}, mpt::MeshPartitionTree)
    return convert_arguments(LS, materialize_edges.(mpt.leaf_meshes))
end

function convert_arguments(M::Type{<:GLMakieMesh}, mpt::MeshPartitionTree)
    return convert_arguments(M, materialize_faces.(mpt.leaf_meshes))
end
