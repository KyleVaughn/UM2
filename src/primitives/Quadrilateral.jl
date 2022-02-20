# Specialized methods for a Quadrilateral, aka Polygon{4}
(quad::Quadrilateral)(r, s) = Point(((1 - r)*(1 - s))quad[1] + (r*(1 - s))quad[2] + 
                                                (r*s)quad[3] + ((1 - r)*s)quad[4])
# This performs much better than the default routine, which is logically equivalent.
# Better simd this way? Chaining isleft doesn't have the same performance improvement for
# triangles.
function Base.in(p::Point2D, quad::Quadrilateral2D)
    return isleft(p, LineSegment2D(quad[1], quad[2])) &&
           isleft(p, LineSegment2D(quad[2], quad[3])) &&
           isleft(p, LineSegment2D(quad[3], quad[4])) &&
           isleft(p, LineSegment2D(quad[4], quad[1]))
end
