function convert_arguments(S::Type{<:Scatter}, mesh::TriMesh)
    return convert_arguments(S, mesh.vertices) 
end

function convert_arguments(LS::Type{<:LineSegments}, mesh::TriMesh)
    return convert_arguments(LS, edges(mesh))
end

function convert_arguments(M::Type{<:GLMakieMesh}, mesh::TriMesh)
    return convert_arguments(M, faces(mesh))
end
