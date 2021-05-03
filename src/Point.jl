import Base: +, -, *, /, ≈, eps

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
Base.firstindex(p::Point) = 1
Base.lastindex(p::Point) = 3
function Base.getindex(p::Point, i::Int)
    1 ≤ i ≤ 3 || throw(BoundsError(p, i))
    if i == 1
        return p.x
    elseif i == 2
        return p.y
    else
        return p.z
    end
end

# Operators
# -------------------------------------------------------------------------------------------------
eps(::Point) = 1.0e-6
≈(p₁::Point, p₂::Point) = (isapprox(p₁.x, p₂.x, atol=eps(p₁)) && 
                           isapprox(p₁.y, p₂.y, atol=eps(p₁)) && 
                           isapprox(p₁.z, p₂.z, atol=eps(p₁))) 
+(p₁::Point, p₂::Point) = Point(p₁.x + p₂.x, p₁.y + p₂.y, p₁.z + p₂.z)
-(p₁::Point, p₂::Point) = Point(p₁.x - p₂.x, p₁.y - p₂.y, p₁.z - p₂.z)
# Cross product for points as vectors
×(p₁::Point, p₂::Point) = Point(p₁.y*p₂.z - p₂.y*p₁.z, 
                                p₁.z*p₂.x - p₂.z*p₁.x, 
                                p₁.x*p₂.y - p₂.x*p₁.y, 
                                )
# Dot product for points as vectors
⋅(p₁::Point, p₂::Point) = p₁.x*p₂.x + p₁.y*p₂.y + p₁.z*p₂.z
+(p::Point, n::Number) = Point(p.x + n, p.y + n, p.z + n)
-(p::Point, n::Number) = Point(p.x - n, p.y - n, p.z - n)
*(n::Number, p::Point) = Point(n*p.x, n*p.y, n*p.z)
/(p::Point, n::Number) = Point(p.x/n, p.y/n, p.z/n)

# Methods
# -------------------------------------------------------------------------------------------------
function distance(p₁::Point, p₂::Point)
  return √( (p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2 + (p₁.z - p₂.z)^2 )
end
