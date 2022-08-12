export face_connectivity

function face_connectivity(mesh::VolumeMesh{2, T, U}) where {T, U}
    if ishomogeneous(mesh)
        facetype = typeof_face(1, mesh)
        nelems = nelements(mesh)
        faces = Vector{facetype}(undef, nelems)
        faces[1:nelems] = reinterpret(facetype, mesh.connectivity)
        return faces
    else
        return map(i -> _materialize_face_connectivity(i, mesh), 1:nelements(mesh))
    end
end

# Not type-stable
function _materialize_face_connectivity(i::Integer, mesh::VolumeMesh{2})
    offset = mesh.offsets[i]
    Δ = offset_diff(i, mesh)
    conn = mesh.connectivity
    if Δ == 3
        tri_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(3))...)]
        return Triangle(tri_vids)
    elseif Δ == 4
        quad_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(4))...)]
        return Quadrilateral(quad_vids)
    elseif Δ == 6
        tri6_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(6))...)]
        return QuadraticTriangle(tri6_vids)
    elseif Δ == 8
        quad8_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(8))...)]
        return QuadraticQuadrilateral(quad8_vids)
    else
        error("Unsupported type.")
        return nothing
    end
end
