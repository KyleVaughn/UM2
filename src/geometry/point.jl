export Point,
       Point2,
       Point2d,
       Point3,
       EPS_POINT,
       EPS_POINT2,
       INF_POINT

export distance2, 
       distance, 
       midpoint,
       isCCw

# Points separated by 1e-5 cm = 0.1 micron are treated the same.
const EPS_POINT = 1e-5
const EPS_POINT2 = EPS_POINT * EPS_POINT

# Used as a sentinel value for infinite points.
const INF_POINT = 1e10

# POINT     
# ---------------------------------------------------------------------------    
#    
# Alias for a vector    
#

const Point = Vec

# -- Type aliases --

const Point2 = Vec{2}
const Point2d = Point2{Float64}
const Point3 = Vec{3}

# -- Methods --

distance2(A::Point, B::Point) = norm2(A - B)
distance(A::Point, B::Point) = norm(A - B)
midpoint(A::Point, B::Point) = (A + B) / 2
isCCW(A::Point2, B::Point2, C::Point2) = 0 < (B - A) × (C - A)
function Base.isapprox(A::Point{D, T}, B::Point{D, T}) where {D, T}
    return distance2(A, B) < T(EPS_POINT2)
end
