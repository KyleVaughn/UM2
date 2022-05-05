# Intersect
# ---------------------------------------------------------------------------------------------
#
# Intersection of a line segment and polygon in 2D
function intersect(l::LineSegment2D{T}, poly::Polygon{N, 2, T}) where {N,T} 
    # Create the line segments that make up the polygon and intersect each one
    # until 2 unique points have been found
    p₁ = nan(Point2D{T}) 
    npoints = 0x0000
    for i ∈ 1:N-1
        hit, point = l ∩ LineSegment2D(poly[i], poly[i + 1]) 
        if hit 
            if npoints === 0x0000 
                npoints = 0x0001
                p₁ = point
            elseif !(p₁ ≈ point)
                return true, SVector(p₁, point)
            end
        end
    end 
    hit, point = l ∩ LineSegment2D(poly[N], poly[1]) 
    if hit 
        if npoints === 0x0000 
            npoints = 0x0001
            p₁ = point
        elseif !(p₁ ≈ point)
            return true, SVector(p₁, point)
        end
    end 
    return false, SVector(p₁, point)
end



# Möller, T., & Trumbore, B. (1997). Fast, minimum storage ray-triangle intersection.
function intersect(l::LineSegment3D{T}, tri::Triangle3D{T}) where {T}
    p = Point3D{T}(0, 0, 0)
    𝗲₁ = tri[2] - tri[1]
    𝗲₂ = tri[3] - tri[1]
    𝗱 = l.𝘂
    𝗽 = 𝗱 × 𝗲₂
    det = 𝗽 ⋅ 𝗲₁
    (det > -1e-8 && det < 1e-8) && return (false, p) 
    inv_det = 1/det
    𝘁 = l.𝘅₁ - tri[1]
    u = (𝘁 ⋅ 𝗽)*inv_det
    (u < 0 || u > 1) && return (false, p)
    𝗾 = 𝘁 × 𝗲₁
    v = (𝗾 ⋅ 𝗱)*inv_det
    (v < 0 || u + v > 1) && return (false, p)
    t = (𝗾 ⋅ 𝗲₂)*inv_det
    (t < 0 || t > 1) && return (false, p)
    return (true, l(t))
end
