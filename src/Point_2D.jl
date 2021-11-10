# @code_warntype checked 2021/11/08

# A 2D point in Cartesian coordinates.
struct Point_2D{T <: AbstractFloat}
    x::SVector{2,T}
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
                                           X <: Real,
                                           Y <: Real} = Point_2D(SVector(T(x),T(y)))
# 1D single conversion constructor
Point_2D(type::Type{T}, x::X) where {T <: AbstractFloat,
                                     X <: Real} = Point_2D(SVector(T(x),T(0)))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p::Point_2D) = Ref(p)
Base.zero(::Point_2D{T}) where {T <: AbstractFloat} = Point_2D((T(0), T(0)))
Base.firstindex(::Point_2D) = 1
Base.lastindex(::Point_2D) = 2
Base.getindex(p::Point_2D, i::Int) = p.x[i]
(::Type{T})(p::Point_2D) where {T <: AbstractFloat} = Point_2D(T.(p.x))

# Operators
# -------------------------------------------------------------------------------------------------
==(p₁::Point_2D, p₂::Point_2D) = (p₁.x == p₂.x)
function ≈(p₁::Point_2D{T}, p₂::Point_2D{T}) where {T <: AbstractFloat}
    return distance(p₁, p₂) < 5e-6
end
+(p₁::Point_2D, p₂::Point_2D) = Point_2D(p₁.x[1] + p₂.x[1], p₁.x[2] + p₂.x[2])
-(p₁::Point_2D, p₂::Point_2D) = Point_2D(p₁.x[1] - p₂.x[1], p₁.x[2] - p₂.x[2])
# Note the cross product of two 2D points returns a scalar. It is assumed that this is the
# desired quantity, since the cross product of vectors in the plane is a vector normal to the plane.
# Hence the z coordinate of the resultant vector is returned.
×(p₁::Point_2D, p₂::Point_2D) = p₁.x[1]*p₂.x[2] - p₂.x[1]*p₁.x[2]
⋅(p₁::Point_2D, p₂::Point_2D) = p₁.x[1]*p₂.x[1] + p₁.x[2]*p₂.x[2]
+(p::Point_2D, n::Real) = Point_2D(p.x[1] + n, p.x[2] + n)
+(n::Real, p::Point_2D) = p + n
-(p::Point_2D, n::Real) = Point_2D(p.x[1] - n, p.x[2] - n)
-(n::Real, p::Point_2D) = p - n
*(n::Real, p::Point_2D) = Point_2D(n*p.x[1], n*p.x[2])
*(p::Point_2D, n::Real) = n*p
/(p::Point_2D, n::Real) = Point_2D(p.x[1]/n, p.x[2]/n)
-(p::Point_2D) = -1*p
# SMatrix multiplication, returns a point
*(A::SMatrix{2, 2, T, 4}, p::Point_2D{T}) where {T <: AbstractFloat} = Point_2D(A * p.x)

# Methods
# -------------------------------------------------------------------------------------------------
# note: hypot is the Julia recommended way to do sqrt of sum squared for 2 numbers
norm(p::Point_2D) = hypot(p.x[1], p.x[2])
function distance(p₁::Point_2D{T}, p₂::Point_2D{T}) where {T <: AbstractFloat}
    return hypot(p₁.x[1] - p₂.x[1], p₁.x[2] - p₂.x[2])
end
midpoint(p₁::Point_2D{T}, p₂::Point_2D{T}) where {T <: AbstractFloat} = (p₁ + p₂)/2

# Plot
# -------------------------------------------------------------------------------------------------
convert_arguments(P::Type{<:Scatter}, p::Point_2D) = convert_arguments(P, p.x)
function convert_arguments(P::Type{<:Scatter}, AP::AbstractArray{<:Point_2D})
    return convert_arguments(P, [p.x for p in AP])
end
