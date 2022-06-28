export Plane

"""
    Plane(𝗻̂::Vec{Dim,T}, d::T)
    Plane(P₁::Point{3}, P₂::Point{3}, P₃::Point{3})

A `Plane` in 3-dimensional space that satisfies X ⋅𝗻̂ = d, where 
X is a `Dim`-dimensional point and 𝗻̂ is the unit normal to the plane.
"""
struct Plane{T}
    𝗻̂::Vec{3, T}
    d::T
end

function Plane(P₁::Point{3}, P₂::Point{3}, P₃::Point{3})
    𝗻̂ = normalize((P₂ - P₁) × (P₃ - P₁))
    return Plane(𝗻̂, coordinates(P₁) ⋅ 𝗻̂)
end
