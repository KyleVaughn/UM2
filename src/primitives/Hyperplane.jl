# A plane in constant-normal form 𝗻 ⋅𝘅 = d
struct Hyperplane{Dim, T}
    𝗻::SVector{Dim, T}
    d::T
end

const Hyperplane2D = Hyperplane{2}
const Hyperplane3D = Hyperplane{3}

# Constructors
# ---------------------------------------------------------------------------------------------
function Hyperplane(a::Point3D, b::Point3D, c::Point3D)

    𝗻 = normalize((b - a) × (c - a))
    isnan(𝗻[1]) && error("Points are collinear") 
    return Hyperplane(𝗻, a.coord ⋅ 𝗻) 
end

function Hyperplane(a::Point2D, b::Point2D)
    𝗻 = normalize(SVector(a[2]-b[2], b[1]-a[1]))
    isnan(𝗻[1]) && error("Points are not unique") 
    return Hyperplane(𝗻, a.coord ⋅ 𝗻) 
end

# Methods 
# ---------------------------------------------------------------------------------------------
Base.in(p::Point, plane::Hyperplane) = p.coord ⋅ plane.𝗻 ≈ plane.d
in_halfspace(p::Point, plane::Hyperplane) = p.coord ⋅ plane.𝗻 - plane.d ≥ 0
# Section 5.3.1 in Ericson, C. (2004). Real-time collision detection
function intersect(l::LineSegment{Dim, T}, plane::Hyperplane{Dim, T}) where {Dim, T}
    r = (plane.d - (plane.𝗻 ⋅ l.𝘅₁.coord))/(plane.𝗻 ⋅l.𝘂)
    (r ≥ 0 && r ≤ 1) && return true, l(r)
    return false, nan_point(Point{Dim,T}) 
end
