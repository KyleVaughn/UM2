export norm2,
       reflect,
       distance2,
       distance,
       midpoint

# -- Vectors --

norm2(v::Vec) = v ⋅ v
reflect(v::Vec, n::Vec) = v - (2 * (v ⋅ n)) * n

# -- Points --

distance2(A::Point, B::Point) = norm2(A - B)
distance(A::Point, B::Point) = sqrt(distance2(A, B))
midpoint(A::Point, B::Point) = (A + B) / 2
function Base.isapprox(A::Point{D, T}, B::Point{D, T}) where {D, T}    
    return distance2(A, B) < T(EPS_POINT2)    
end                                                                      
isCCW(A::Point{2}, B::Point{2}, C::Point{2}) = 0 ≤ (B - A) × (C - A)
