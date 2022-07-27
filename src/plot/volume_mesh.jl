function convert_arguments(S::Type{<:Scatter}, mesh::VolumeMesh)
  return convert_arguments(S, points(mesh))
end

function convert_arguments(LS::Type{<:LineSegments}, mesh::VolumeMesh{2})
    return convert_arguments(LS, materialize_edges(mesh))
end

function mesh!(mesh::VolumeMesh{2})
    mesh_faces = materialize_faces(mesh)
    unique_mats = unique(mesh.materials)
    for mat in unique_mats
        mesh!(mesh_faces[map(m->m == mat, mesh.materials)])
    end
    return nothing
end
