# A 3D point in Cartesian coordinates.
import Base: +, -, *, /, ≈, ==
import LinearAlgebra: norm

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
==(p⃗₁::Point, p⃗₂::Point) = (p⃗₁.coord == p⃗₂.coord)
function ≈(p⃗₁::Point{T}, p⃗₂::Point{T}) where {T <: AbstractFloat}
    # If non-zero, use relative isapprox. If zero, use absolute isapprox.
    # Otherwise, x ≈ 0 is false for every x ≢ 0
    bool = [true, true, true]
    for i = 1:3
        if (p⃗₁[i] == T(0)) || (p⃗₂[i] == T(0))
            bool[i] = isapprox(p⃗₁[i], p⃗₂[i], atol=sqrt(eps(T)))
        else
            bool[i] = isapprox(p⃗₁[i], p⃗₂[i])
        end
    end
    return all(bool)
end
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
+(n::Number, p⃗::Point) = p⃗ + n
-(p⃗::Point, n::Number) = Point(p⃗.coord .- n)
-(n::Number, p⃗::Point) = p⃗ - n
*(n::Number, p⃗::Point) = Point(p⃗.coord .* n)
*(p⃗::Point, n::Number) = n*p⃗
/(p⃗::Point, n::Number) = Point(p⃗.coord ./ n)
# Unary -
-(p⃗::Point) = -1*p⃗

# Methods
# -------------------------------------------------------------------------------------------------
"""
    distance(p⃗₁::Point, p⃗₂::Point)

Returns the Euclidian distance from `p⃗₁` to `p⃗₂`.
"""
function distance(p⃗₁::Point, p⃗₂::Point)
    return √( (p⃗₁[1] - p⃗₂[1])^2 + (p⃗₁[2] - p⃗₂[2])^2 + (p⃗₁[3] - p⃗₂[3])^2 )
end

function norm(p⃗::Point)
    return norm(p⃗.coord)
end
