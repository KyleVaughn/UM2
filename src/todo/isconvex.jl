isconvex(tri::Triangle2D) = true
function isconvex(poly::Polygon{N, 2}) where {N} 
    # If each of the point triplets in the polygon is CCW oriented, it is convex
    vec₁ = poly[2] - poly[1]
    for i ∈ 1:N-2
        vec₂ = poly[i+2] - poly[i+1]
        vec₁ × vec₂ < 0 && return false
        vec₁ = vec₂
    end 
    return 0 ≤ vec₁ × (poly[1] - poly[N])
end
