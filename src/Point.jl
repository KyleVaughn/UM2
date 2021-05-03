import Base: +, -, *, /, ≈

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

# Base methods
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p::Point) = Ref(p)
Base.zero(::Point{T}) where {T<:AbstractFloat} = Point(zero(T), zero(T), zero(T))

# Operators
# -------------------------------------------------------------------------------------------------
≈(p1::Point, p2::Point) = (p1.x ≈ p2.x) && (p1.y ≈ p2.y) && (p1.z ≈ p2.z)
+(p1::Point, p2::Point) = Point(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z)
-(p1::Point, p2::Point) = Point(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
+(p::Point, n::Number) = Point(p.x + n, p.y + n, p.z + n)
-(p::Point, n::Number) = Point(p.x - n, p.y - n, p.z - n)
*(n::Number, p::Point) = Point(n*p.x, n*p.y, n*p.z)
/(p::Point, n::Number) = Point(p.x/n, p.y/n, p.z/n)

# Methods
# -------------------------------------------------------------------------------------------------
function distance(p1::Point, p2::Point)
  return √( (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2 )
end
