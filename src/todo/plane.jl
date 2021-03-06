export Plane

"""
    Plane(š»Ģ::Vec{Dim,T}, d::T)
    Plane(Pā::Point{3}, Pā::Point{3}, Pā::Point{3})

A `Plane` in 3-dimensional space that satisfies X āš»Ģ = d, where 
X is a `Dim`-dimensional point and š»Ģ is the unit normal to the plane.
"""
struct Plane{T}
    š»Ģ::Vec{3, T}
    d::T
end

function Plane(Pā::Point{3}, Pā::Point{3}, Pā::Point{3})
    š»Ģ = normalize((Pā - Pā) Ć (Pā - Pā))
    return Plane(š»Ģ, coordinates(Pā) ā š»Ģ)
end
