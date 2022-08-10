# Section 5.3.1 in Ericson, C. (2004). Real-time collision detection
function intersect(l::LineSegment{Dim, T}, plane::Hyperplane{Dim, T}) where {Dim, T}
    r = (plane.d - (plane.𝗻 ⋅ l.𝘅₁.coord)) / (plane.𝗻 ⋅ l.𝘂)
    (r ≥ 0 && r ≤ 1) && return true, l(r)
    return false, nan(Point{Dim, T})
end
