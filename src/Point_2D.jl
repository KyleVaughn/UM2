# A 2D point in Cartesian coordinates.

struct Point_2D{T <: AbstractFloat}
    coord::SVector{2,T}
end

# Constructors
# -------------------------------------------------------------------------------------------------
# 2D single constructor
Point_2D(x::T, y::T) where {T <: AbstractFloat} = Point_2D(SVector(x, y))
# 1D single constructor
Point_2D(x::T) where {T <: AbstractFloat} = Point_2D(SVector(x, T(0)))
# 2D tuple constructor
Point_2D((x, y)::Tuple{T,T}) where {T <: AbstractFloat} = Point_2D(SVector(x, y))
# 2D single conversion constructor
Point_2D(type::Type{T}, x::X, y::Y) where {T <: AbstractFloat,
                                       X,Y <: Real} = Point_2D(SVector(T(x),T(y)))
# 1D single conversion constructor
Point_2D(type::Type{T}, x::X) where {T <: AbstractFloat,
                                 X <: Real} = Point_2D(SVector(T(x),T(0)))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p⃗::Point_2D) = Ref(p⃗)
Base.zero(::Point_2D{T}) where {T <: AbstractFloat} = Point_2D((T(0), T(0)))
Base.firstindex(::Point_2D) = 1
Base.lastindex(::Point_2D) = 2
Base.getindex(p⃗::Point_2D, i::Int) = p⃗.coord[i]
(::Type{T})(p⃗::Point_2D) where {T <: AbstractFloat} = Point_2D(T.(p⃗.coord))

# Operators
# -------------------------------------------------------------------------------------------------
==(p⃗₁::Point_2D, p⃗₂::Point_2D) = (p⃗₁.coord == p⃗₂.coord)
function ≈(p⃗₁::Point_2D{T}, p⃗₂::Point_2D{T}) where {T <: AbstractFloat}
    # If non-zero, use relative isapprox. If zero, use absolute isapprox.
    # Otherwise, x ≈ 0 is false for every x ≠ 0
    bool = [true, true]
    if (p⃗₁[1] == T(0)) || (p⃗₂[1] == T(0))
        bool[1] = isapprox(p⃗₁[1], p⃗₂[1], atol=sqrt(eps(T)))
    else
        bool[1] = isapprox(p⃗₁[1], p⃗₂[1])
    end
    if (p⃗₁[2] == T(0)) || (p⃗₂[2] == T(0))
        bool[2] = isapprox(p⃗₁[2], p⃗₂[2], atol=sqrt(eps(T)))
    else
        bool[2] = isapprox(p⃗₁[2], p⃗₂[2])
    end
    return all(bool)
end
+(p⃗₁::Point_2D, p⃗₂::Point_2D) = Point_2D(p⃗₁.coord[1] + p⃗₂.coord[1], p⃗₁.coord[2] + p⃗₂.coord[2])
-(p⃗₁::Point_2D, p⃗₂::Point_2D) = Point_2D(p⃗₁.coord[1] - p⃗₂.coord[1], p⃗₁.coord[2] - p⃗₂.coord[2])
×(p⃗₁::Point_2D, p⃗₂::Point_2D) = p⃗₁.coord[1]*p⃗₂.coord[2] - p⃗₂.coord[1]*p⃗₁.coord[2]
⋅(p⃗₁::Point_2D, p⃗₂::Point_2D) = p⃗₁.coord[1]*p⃗₂.coord[1] + p⃗₁.coord[2]*p⃗₂.coord[2]
+(p⃗::Point_2D, n::Real) = Point_2D(p⃗.coord[1] + n, p⃗.coord[2] + n)
+(n::Real, p⃗::Point_2D) = p⃗ + n
-(p⃗::Point_2D, n::Real) = Point_2D(p⃗.coord[1] - n, p⃗.coord[2] - n)
-(n::Real, p⃗::Point_2D) = p⃗ - n
*(n::Real, p⃗::Point_2D) = Point_2D(n*p⃗.coord[1], n*p⃗.coord[2])
*(p⃗::Point_2D, n::Real) = n*p⃗
/(p⃗::Point_2D, n::Real) = Point_2D(p⃗.coord[1]/n, p⃗.coord[2]/n)
-(p⃗::Point_2D) = -1*p⃗

# Methods
# -------------------------------------------------------------------------------------------------
norm(p⃗::Point_2D) = hypot(p⃗.coord[1], p⃗.coord[2])
distance(p⃗₁::Point_2D, p⃗₂::Point_2D) = norm(p⃗₁ - p⃗₂)

# Plot
# -------------------------------------------------------------------------------------------------
convert_arguments(P::Type{<:Scatter}, p::Point_2D) = convert_arguments(P, p.coord)
function convert_arguments(P::Type{<:Scatter}, AP::AbstractArray{<:Point_2D})
    return convert_arguments(P, [p.coord for p in AP])
end
