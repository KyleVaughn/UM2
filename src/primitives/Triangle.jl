# Specialized methods for a Triangle, aka Polygon{3}
(tri::Triangle)(r, s) = Point((1 - r - s)*tri[1] + r*tri[2] + s*tri[3])
area(tri::Triangle2D) = norm((tri[2] - tri[1]) × (tri[3] - tri[1]))/2
area(tri::Triangle3D) = norm((tri[2] - tri[1]) × (tri[3] - tri[1]))/2
centroid(tri::Triangle2D) = Point((tri[1] + tri[2] + tri[3])/3)
centroid(tri::Triangle3D) = Point((tri[1] + tri[2] + tri[3])/3)

# Point inside polygon
# ---------------------------------------------------------------------------------------------
function Base.in(p::Point3D, tri::Triangle3D)
    # Translate triangle coordinate system such that p is the origin
    𝗮 = tri[1] - p
    𝗯 = tri[2] - p
    𝗰 = tri[3] - p
    # We may check that p is within the 3 half-spaces that bound the triangle ABC
    # by ensuring that PAB, PBC, and PCA are all counter-clockwise oriented. 
    # This can be tested by ensuring that 𝗯 × 𝗰, 𝗰 × 𝗮, 𝗮 × 𝗯 all have the same sign.
    # C
    # | \
    # |   \
    # |   P \
    # |       \
    # A---------B
    𝘂 = 𝗯 × 𝗰 
    𝘃 = 𝗰 × 𝗮
    𝘂 ⋅ 𝘃 < 0 && return false
    𝘄 = 𝗮 × 𝗯
    𝘂 ⋅ 𝘄 < 0 && return false
    # If we have reached this point, p is within the 3 half-spaces that bound ABC.
    # To ensure that p is on the same plane as ABC, and hence in ABC, we check that
    # the sign of each component of the normals is the same.
    #
    #
    # IS it faster just to compute the normals and check it all out the gate?
    # that would def be branchless.
end

# Intersect
# ---------------------------------------------------------------------------------------------
function intersect(l::LineSegment2D{T}, tri::Triangle2D{T}
                  ) where {T <: Union{Float32, Float64}} 
    # Options are 0, 2, or 3 intersections
    hit₁, p₁ = l ∩ LineSegment2D(tri[1], tri[2])
    hit₂, p₂ = l ∩ LineSegment2D(tri[2], tri[3])
    hit₃, p₃ = l ∩ LineSegment2D(tri[3], tri[1])
    # Possibilities: 1+2, 1+3, 2+3, 1+2+3, none. 
    if hit₁
        if hit₂
            if hit₃ # 1+2+3
                return 0x0003, SVector(p₁, p₂, p₃)
            else # 1+2
                return 0x0002, SVector(p₁, p₂, p₃)
            end
        else # 1+3
            return 0x0002, SVector(p₁, p₃, p₂)
        end
    elseif hit₂ # 2+3
        return 0x0002, SVector(p₂, p₃, p₁)
    else # none
        return 0x0000, SVector(p₁, p₂, p₃)
    end
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
