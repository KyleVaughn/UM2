import Base: +, -, *, /, ≈

struct Point{T <: AbstractFloat}
    coord::NTuple{3,T}
end

# Constructors
# -------------------------------------------------------------------------------------------------
# 3D single constructor
Point(x,y,z) = Point((x,y,z))
# 2D constructor
Point((x, y)) = Point((x, y, zero(x)))
# 2D single constructor
Point(x, y) = Point((x, y, zero(x)))

# Base methods
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p⃗::Point) = Ref(p⃗)
Base.zero(::Point{T}) where {T <: AbstractFloat} = Point((zero(T), zero(T), zero(T)))
Base.firstindex(::Point) = 1
Base.lastindex(::Point) = 3
Base.getindex(p⃗::Point, i::Int) = p⃗.coord[i]

# Operators
# -------------------------------------------------------------------------------------------------
atol(::Point) = 1.0e-6
≈(p⃗₁::Point, p⃗₂::Point) = all(isapprox.(p⃗₁.coord, p⃗₂.coord, atol=atol(p⃗₁)))
+(p⃗₁::Point, p⃗₂::Point) = Point(p⃗₁.coord .+ p⃗₂.coord)
-(p⃗₁::Point, p⃗₂::Point) = Point(p⃗₁.coord .- p⃗₂.coord)
# Cross product
×(p⃗₁::Point, p⃗₂::Point) = Point(p⃗₁[2]*p⃗₂[3] - p⃗₂[2]*p⃗₁[3], 
                                p⃗₁[3]*p⃗₂[1] - p⃗₂[3]*p⃗₁[1], 
                                p⃗₁[1]*p⃗₂[2] - p⃗₂[1]*p⃗₁[2], 
                                )
# Dot product
⋅(p⃗₁::Point, p⃗₂::Point) = p⃗₁[1]*p⃗₂[1] + p⃗₁[2]*p⃗₂[2] + p⃗₁[3]*p⃗₂[3]
+(p⃗::Point, n::Number) = Point(p⃗.coord .+ n)
-(p⃗::Point, n::Number) = Point(p⃗.coord .- n)
*(n::Number, p⃗::Point) = Point(p⃗.coord .* n)
/(p⃗::Point, n::Number) = Point(p⃗.coord ./ n)

# Methods
# -------------------------------------------------------------------------------------------------
"""
    distance(p⃗₁::Point, p⃗₂::Point)

Returns the Euclidian distance from `p⃗₁` to `p⃗₂`.
"""
function distance(p⃗₁::Point, p⃗₂::Point)
    return √( (p⃗₁[1] - p⃗₂[1])^2 + (p⃗₁[2] - p⃗₂[2])^2 + (p⃗₁[3] - p⃗₂[3])^2 )
end
