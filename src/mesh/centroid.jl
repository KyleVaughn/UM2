export centroid

function centroid(i::Integer, mesh::VolumeMesh{2, T}) where {T}
    conn = mesh.connectivity
    offset = mesh.offsets[i]
    Δ = offset_diff(i, mesh)
    if Δ == 3
        tri_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(3))...)]
        return centroid(materialize(Triangle(tri_vids), mesh.points))
    elseif Δ == 4
        quad_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(4))...)]
        return centroid(materialize(Quadrilateral(quad_vids), mesh.points))
    elseif Δ == 6
        tri6_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(6))...)]
        return centroid(materialize(QuadraticTriangle(tri6_vids), mesh.points))
    elseif Δ == 8
        quad8_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(8))...)]
        return centroid(materialize(QuadraticQuadrilateral(quad8_vids), mesh.points))
    else
        error("Unsupported type.")
        return zero(T)
    end
end

function centroid(i::Integer, mesh::PolytopeVertexMesh{2})
  return centroid(materialize(mesh.polytopes[i], mesh.vertices))
end
