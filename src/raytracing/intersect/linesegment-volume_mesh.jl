function Base.intersect(l::LineSegment{Point{2, T}}, 
                        mesh::VolumeMesh{2, T, U}) where {T, U} 
    P₀ = l[1]
    intersections = Point{2, T}[]
    conn = mesh.connectivity
    for i in 1:nelements(mesh)
        offset = mesh.offsets[i]
        Δ = offset_diff(i, mesh)
        if Δ == 3
            tri_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(3))...)]
            tri_intersections = l ∩ materialize(Triangle(tri_vids), 
                                                mesh.points)
            _insert_valid_intersections!(intersections, P₀, tri_intersections)
        elseif Δ == 4
            quad_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(4))...)]
            quad_intersections = l ∩ materialize(Quadrilateral(quad_vids), 
                                                 mesh.points)
            _insert_valid_intersections!(intersections, P₀, quad_intersections)
        elseif Δ == 6
            tri6_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(6))...)]
            tri6_intersections = l ∩ materialize(QuadraticTriangle(tri6_vids),
                                                 mesh.points)
            _insert_valid_intersections!(intersections, P₀, tri6_intersections)
        elseif Δ == 8
            quad8_vids = conn[Vec(ntuple(i -> i + offset - 1, Val(8))...)]
            quad8_intersections = l ∩ materialize(QuadraticQuadrilateral(quad8_vids),
                                                  mesh.points)
            _insert_valid_intersections!(intersections, P₀, quad8_intersections)
        else
            error("Unsupported type.")
            return nothing
        end 
    end
    return intersections 
end
