export materialize,
#       materialize_edge,
       materialize_face,
       materialize_cell,
       materialize_polytope,
       materialize_edges,
       materialize_faces,
       materialize_cells,
       materialize_polytopes

function materialize(p::Polytope{K, P, N, T},
                     vertices::Vector{Point{D, F}}) where {K, P, N, T, D, F}
    return Polytope{K, P, N, Point{D, F}}(Vec(ntuple(i -> vertices[p.vertices[i]],
                                                     Val(N))))
end

# Volume materialization
function materialize_edges(mesh::VolumeMesh{2})
    return materialize.(edge_connectivity(mesh), Ref(mesh.points))
end

function materialize_face(i::Integer, mesh::VolumeMesh{2})
    return materialize(_materialize_face_connectivity(i, mesh), mesh.points)
end

function materialize_faces(mesh::VolumeMesh{2})
    return materialize.(face_connectivity(mesh), Ref(mesh.points))
end

# Polytope materialization
function materialize_edges(mesh::PolytopeVertexMesh{2})
    return materialize.(edge_connectivity(mesh), Ref(mesh.vertices))
end

function materialize_polytope(i::Integer, mesh::PolytopeVertexMesh)
    return materialize(mesh.polytopes[i], mesh.vertices)
end

function materialize_polytopes(mesh::PolytopeVertexMesh)
    return materialize.(mesh.polytopes, Ref(mesh.vertices))
end

# aliases
function materialize_polytope(i::Integer, mesh::VolumeMesh{2})
    return materialize_face(i, mesh)
end

function materialize_polytopes(mesh::VolumeMesh{2})
    return materialize_faces(mesh)
end

function materialize_faces(mesh::PolytopeVertexMesh{D, T, P}) where {D, T, P <: Face}
    return materialize_polytopes(mesh)
end

function materialize_cells(mesh::PolytopeVertexMesh{D, T, P}) where {D, T, P <: Cell}
    return materialize_polytopes(mesh)
end

function materialize_facets(mesh::PolytopeVertexMesh{D, T, P}) where {D, T, P <: Cell}
    return materialize_faces(mesh)
end

function materialize_ridges(mesh::PolytopeVertexMesh{D, T, P}) where {D, T, P <: Cell}
    return materialize_edges(mesh)
end

function materialize_facets(mesh::PolytopeVertexMesh{D, T, P}) where {D, T, P <: Face}
    return materialize_edges(mesh)
end
