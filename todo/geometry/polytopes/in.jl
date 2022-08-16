# Test if a point is in a polygon for 2D points/polygons
function Base.in(P::Point{2, T}, poly::Polygon{N, Point{2, T}}) where {N, T}
    for i in Base.OneTo(N - 1)
        isleft(P, LineSegment(poly[i], poly[i + 1])) || return false
    end
    return isleft(P, LineSegment(poly[N], poly[1]))
end

# Test if a point is in a polygon for 2D points/quadratic polygons
function Base.in(P::Point{2, T}, poly::QuadraticPolygon{N, Point{2, T}}) where {N, T}
    M = N ÷ 2
    for i in Base.OneTo(M - 1)
        isleft(P, QuadraticSegment(poly[i], poly[i + 1], poly[i + M])) || return false
    end
    return isleft(P, QuadraticSegment(poly[M], poly[1], poly[N]))
end
