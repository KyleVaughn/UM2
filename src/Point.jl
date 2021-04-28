# Constructors
# -------------------------------------------------------------------------------------------------
struct Point{T<:AbstractFloat}
  x::T
  y::T
  z::T
end
# 2D point constructor. z = 0
Point(x, y) = Point(x, y, zero(x))
# 1D point constructor. y = z = 0
Point(x) = Point(x, zero(x), zero(x))

# Methods
# -------------------------------------------------------------------------------------------------
function distance(p1::Point, p2::Point)
  return √( (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2 )
end
