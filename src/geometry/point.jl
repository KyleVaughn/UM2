export Point,
       Point2,
       Point2f,
       Point2d,
       EPS_POINT,
       EPS_POINT2,
       INF_POINT

# Points separated by 1e-5 cm = 0.1 micron are treated the same.
const EPS_POINT = 1e-5
const EPS_POINT2 = EPS_POINT * EPS_POINT

# Used for when IEEE 754 may not be enforced, such as with fast math. 
const INF_POINT = 1e10

# POINT     
# ---------------------------------------------------------------------------    
#    
# Alias for a vector    
#
const Point = Vec

# -- Type aliases --

const Point2 = Vec{2}
const Point2f = Vec2f
const Point2d = Vec2d

# -- Methods --
distance2(a::Point, b::Point) = norm2(a - b)
distance(a::Point, b::Point) = norm(a - b)
midpoint(a::Point, b::Point) = (a + b) / 2
isCCW(a::Point2, b::Point2, c::Point2) = 0 < (b - a) × (c - a)
Base.isapprox(a::Point{D, T}, b::Point{D, T}) where {D, T} = distance2(a, b) < T(EPS_POINT2)
