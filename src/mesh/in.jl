function Base.in(P::Point{2, T}, mesh::VolumeMesh{2, T}) where {T}
    conn = mesh.connectivity
    for i in 1:nelements(mesh)
        offset = mesh.offsets[i]
        Δ = offset_diff(i, mesh)
        if Δ == 3
            tri_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(3))...)]
            bool = P ∈ materialize(Triangle(tri_vids), mesh.points)
        elseif Δ == 4
            quad_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(4))...)]
            bool = P ∈ materialize(Quadrilateral(quad_vids), mesh.points)
        elseif Δ == 6
            tri6_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(6))...)]
            bool = P ∈ materialize(QuadraticTriangle(tri6_vids), mesh.points)
        elseif Δ == 8
            quad8_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(8))...)]
            bool = P ∈ materialize(QuadraticQuadrilateral(quad8_vids), mesh.points)
        else
            error("Unsupported type.")
            return 0 
        end
        bool && return i
    end
    error("Unable to find mesh element containing "*string(P))
    return 0
end

function Base.in(P::Point{2, T}, mesh::PolytopeVertexMesh{2, T}) where {T}
    for i in 1:nelements(mesh) 
        P ∈ materialize(mesh.polytopes[i], mesh.vertices) && return i 
    end
    error("Unable to find mesh element containing "*string(P))
    return 0 
end
