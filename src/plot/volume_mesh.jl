function convert_arguments(S::Type{<:Scatter}, mesh::VolumeMesh)
  return convert_arguments(S, points(mesh))
end

function convert_arguments(LS::Type{<:LineSegments}, mesh::VolumeMesh{2})
    return convert_arguments(LS, materialize_edges(mesh))
end

function convert_arguments(M::Type{<:GLMakieMesh}, mesh::VolumeMesh{2})
    return convert_arguments(M, materialize_faces(mesh))
end
