# A 3D point in Cartesian coordinates.
import Base: +, -, *, /, ≈, ==
import LinearAlgebra: norm
import GLMakie: convert_arguments

struct Point{T <: AbstractFloat}
    coord::SVector{3,T}
end

# Constructors
# -------------------------------------------------------------------------------------------------
# 3D single constructor
Point(x::T, y::T, z::T) where {T <: AbstractFloat} = Point(SVector(x,y,z))
# 2D single constructor
Point(x::T, y::T) where {T <: AbstractFloat} = Point(SVector(x, y, zero(x)))
# 1D single constructor
Point(x::T) where {T <: AbstractFloat} = Point(SVector(x, zero(x), zero(x)))
# 3D tuple constructor
Point(x::T) where {T <: NTuple{3}} = Point(SVector(x))
# 2D tuple constructor
Point((x, y)::T) where {T <: NTuple{2}} = Point(SVector(x, y, zero(x)))

# Base
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
    # Otherwise, x ≈ 0 is false for every x ≠ 0
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
×(p⃗₁::Point, p⃗₂::Point) = Point(p⃗₁[2]*p⃗₂[3] - p⃗₂[2]*p⃗₁[3],
                                p⃗₁[3]*p⃗₂[1] - p⃗₂[3]*p⃗₁[1],
                                p⃗₁[1]*p⃗₂[2] - p⃗₂[1]*p⃗₁[2],
                                )
⋅(p⃗₁::Point, p⃗₂::Point) = p⃗₁[1]*p⃗₂[1] + p⃗₁[2]*p⃗₂[2] + p⃗₁[3]*p⃗₂[3]

+(p⃗::Point, n::Number) = Point(p⃗.coord .+ n)
+(n::Number, p⃗::Point) = p⃗ + n
-(p⃗::Point, n::Number) = Point(p⃗.coord .- n)
-(n::Number, p⃗::Point) = p⃗ - n
*(n::Number, p⃗::Point) = Point(p⃗.coord .* n)
*(p⃗::Point, n::Number) = n*p⃗
/(p⃗::Point, n::Number) = Point(p⃗.coord ./ n)
-(p⃗::Point) = -1*p⃗

# Methods
# -------------------------------------------------------------------------------------------------
norm(p⃗::Point) = norm(p⃗.coord)
distance(p⃗₁::Point, p⃗₂::Point) = norm(p⃗₁ - p⃗₂)

# Plot
# -------------------------------------------------------------------------------------------------
@recipe function plot_point(p::Point)
    return [p[1]], [p[2]], [p[3]]
end
@recipe function plot_point(AP::AbstractArray{<:Point} )
    return map(p->p[1], AP), map(p->p[2], AP), map(p->p[3], AP)
end
convert_arguments(p::Point) = convert_arguments(P, p.coord)
convert_arguments(AP::AbstractArray{<:Point}) = convert_arguments(P, [p.coord for p in AP])
