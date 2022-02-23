"""
    Hyperplane(𝗻̂::SVector{Dim, T}, d::T)
    Hyperplane(a::Point2D, b::Point2D)
    Hyperplane(a::Point3D, b::Point3D, c::Point3D)

Construct a hyperplane in `Dim`-dimensional space that satisfies 𝘅 ⋅𝗻̂ = d, where 
𝘅 is a `Dim`-dimensional point and 𝗻̂ is the unit normal to the plane.
"""
struct Hyperplane{Dim, T}
    𝗻̂::SVector{Dim, T}
    d::T
end

const Hyperplane2D = Hyperplane{2}
const Hyperplane3D = Hyperplane{3}

function Hyperplane(a::Point3D, b::Point3D, c::Point3D)
    𝗻̂ = normalize((b - a) × (c - a))
    return Hyperplane(𝗻̂, a.coord ⋅ 𝗻̂) 
end

function Hyperplane(a::Point2D, b::Point2D)
    𝗻̂ = normalize(SVector(a[2]-b[2], b[1]-a[1]))
    return Hyperplane(𝗻̂, a.coord ⋅ 𝗻̂) 
end
