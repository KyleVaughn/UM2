function convert_arguments(LS::Type{<:LineSegments}, mesh::UnstructuredMesh)
    return convert_arguments(LS, materialize_edges(mesh))
end

function convert_arguments(P::Type{<:Mesh}, mesh::UnstructuredMesh)
    return convert_arguments(P, materialize_faces(mesh))
end
