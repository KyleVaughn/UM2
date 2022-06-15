# Intersection of a line segment and polygon in 2D
function Base.intersect(l::LineSegment{Point{2, T}}, 
                        poly::Polygon{N, Point{2, T}}) where {N, T} 
    # Create the line segments that make up the polygon and intersect each one
    # until 2 unique points have been found
    T_INF_POINT = T(INF_POINT)
    p1 = Point(T_INF_POINT, T_INF_POINT)
    one_hit = false
    for i in Base.OneTo(N)
        pt = l ∩ LineSegment(poly[(i - 1) % N + 1], poly[i % N + 1]) 
        if pt[1] !== T_INF_POINT 
            if !one_hit 
                one_hit = true
                p1 = pt
            elseif p1 ≉ pt
                return Vec(p1, pt)
            end
        end
    end 
    return Vec(p1, Point(T_INF_POINT, T_INF_POINT)) 
end

# # Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray-triangle intersection.
# function intersect(l::LineSegment3D{T}, tri::Triangle3D{T}) where {T}
#     p = Point3D{T}(0, 0, 0)
#     𝗲₁ = tri[2] - tri[1]
#     𝗲₂ = tri[3] - tri[1]
#     𝗱 = l.𝘂
#     𝗽 = 𝗱 × 𝗲₂
#     det = 𝗽 ⋅ 𝗲₁
#     (det > -1e-8 && det < 1e-8) && return (false, p) 
#     inv_det = 1/det
#     𝘁 = l.𝘅₁ - tri[1]
#     u = (𝘁 ⋅ 𝗽)*inv_det
#     (u < 0 || u > 1) && return (false, p)
#     𝗾 = 𝘁 × 𝗲₁
#     v = (𝗾 ⋅ 𝗱)*inv_det
#     (v < 0 || u + v > 1) && return (false, p)
#     t = (𝗾 ⋅ 𝗲₂)*inv_det
#     (t < 0 || t > 1) && return (false, p)
#     return (true, l(t))
# end
