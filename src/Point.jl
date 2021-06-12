# A 3D point in Cartesian coordinates.

struct Point{T <: AbstractFloat}
    coord::SVector{3,T}
end

# Constructors
# -------------------------------------------------------------------------------------------------
# 3D single constructor
Point(x::T, y::T, z::T) where {T <: AbstractFloat} = Point(SVector(x,y,z))
# 2D single constructor
Point(x::T, y::T) where {T <: AbstractFloat} = Point(SVector(x, y, T(0)))
# 1D single constructor
Point(x::T) where {T <: AbstractFloat} = Point(SVector(x, T(0), T(0)))
# 3D tuple constructor
Point(x::Tuple{T,T,T}) where {T <: AbstractFloat} = Point(SVector(x))
# 2D tuple constructor
Point((x, y)::Tuple{T,T}) where {T <: AbstractFloat} = Point(SVector(x, y, T(0)))
# 3D single conversion constructor
Point(type::Type{T}, x::X, y::Y, z::Z) where {T <: AbstractFloat,
                                              X,Y,Z <: Real} = Point(SVector(T(x),T(y),T(z)))
# 2D single conversion constructor
Point(type::Type{T}, x::X, y::Y) where {T <: AbstractFloat,
                                       X,Y <: Real} = Point(SVector(T(x),T(y),T(0)))
# 1D single conversion constructor
Point(type::Type{T}, x::X) where {T <: AbstractFloat,
                                 X <: Real} = Point(SVector(T(x),T(0),T(0)))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p⃗::Point) = Ref(p⃗)
Base.zero(::Point{T}) where {T <: AbstractFloat} = Point((T(0), T(0), T(0)))
Base.firstindex(::Point) = 1
Base.lastindex(::Point) = 3
Base.getindex(p⃗::Point, i::Int) = p⃗.coord[i]
(::Type{T})(p⃗::Point) where {T <: AbstractFloat} = Point(T.(p⃗.coord))

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

+(p⃗::Point, n::Real) = Point(p⃗.coord .+ n)
+(n::Real,  p⃗::Point) = p⃗ + n
-(p⃗::Point, n::Real) = Point(p⃗.coord .- n)
-(n::Real,  p⃗::Point) = p⃗ - n
*(n::Real,  p⃗::Point) = Point(p⃗.coord .* n)
*(p⃗::Point, n::Real) = n*p⃗
/(p⃗::Point, n::Real) = Point(p⃗.coord ./ n)
-(p⃗::Point) = -1*p⃗

# Methods
# -------------------------------------------------------------------------------------------------
norm(p⃗::Point) = norm(p⃗.coord)
distance(p⃗₁::Point, p⃗₂::Point) = norm(p⃗₁ - p⃗₂)

# Plot
# -------------------------------------------------------------------------------------------------
convert_arguments(P::Type{<:Scatter}, p::Point) = convert_arguments(P, p.coord)
function convert_arguments(P::Type{<:Scatter}, AP::AbstractArray{<:Point})
    return convert_arguments(P, [p.coord for p in AP])
end
