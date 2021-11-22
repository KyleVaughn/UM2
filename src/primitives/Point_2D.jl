# @code_warntype checked 2021/11/19

# A 2D point in Cartesian coordinates.
struct Point_2D{F <: AbstractFloat}
    x::SVector{2, F}
end

# Constructors
# -------------------------------------------------------------------------------------------------
# single constructor
# @code_warntype checked 2021/11/19
Point_2D(x::F, y::F) where {F <: AbstractFloat} = Point_2D(SVector(x, y))

# 1D single constructor
# @code_warntype checked 2021/11/19
Point_2D(x::F) where {F <: AbstractFloat} = Point_2D(SVector(x, F(0)))

# tuple constructor
# @code_warntype checked 2021/11/19
Point_2D(t::Tuple{F, F}) where {F <: AbstractFloat} = Point_2D(SVector(t))

# single conversion constructor
# @code_warntype checked 2021/11/19
Point_2D(float_type::Type{F}, x::X, y::Y) where {F <: AbstractFloat,
                                                 X <: Real,
                                                 Y <: Real} = Point_2D(SVector(F(x), F(y)))
# 1D single conversion constructor
# @code_warntype checked 2021/11/19
Point_2D(float_type::Type{F}, x::X) where {F <: AbstractFloat,
                                           X <: Real} = Point_2D(SVector(F(x), F(0)))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p::Point_2D) = Ref(p)
# @code_warntype checked 2021/11/19
Base.zero(::Point_2D{F}) where {F <: AbstractFloat} = Point_2D(F(0), F(0))
# @code_warntype checked 2021/11/19
Base.getindex(p::Point_2D, i::I) where {I <: Integer} = p.x[i]
Base.firstindex(::Point_2D) = 1
Base.lastindex(::Point_2D) = 2

# Operators
# -------------------------------------------------------------------------------------------------
# @code_warntype checked 2021/11/19
==(p₁::Point_2D, p₂::Point_2D) = (p₁.x == p₂.x)
# @code_warntype checked 2021/11/19
function ≈(p₁::Point_2D, p₂::Point_2D)
    return distance(p₁, p₂) < Point_2D_differentiation_distance
end
# @code_warntype checked 2021/11/19
≉(p₁::Point_2D, p₂::Point_2D) = !(p₁ ≈ p₂)
# @code_warntype checked 2021/11/19
+(p₁::Point_2D, p₂::Point_2D) = Point_2D(p₁.x[1] + p₂.x[1], p₁.x[2] + p₂.x[2])
# @code_warntype checked 2021/11/19
-(p₁::Point_2D, p₂::Point_2D) = Point_2D(p₁.x[1] - p₂.x[1], p₁.x[2] - p₂.x[2])
# Note the cross product of two 2D points returns a scalar. It is assumed that this is the
# desired quantity, since the cross product of vectors in the plane is a vector normal to the plane.
# Hence the z coordinate of the resultant vector is returned.
# @code_warntype checked 2021/11/19
×(p₁::Point_2D, p₂::Point_2D) = p₁.x[1]*p₂.x[2] - p₂.x[1]*p₁.x[2]
# @code_warntype checked 2021/11/19
⋅(p₁::Point_2D, p₂::Point_2D) = p₁.x[1]*p₂.x[1] + p₁.x[2]*p₂.x[2]
# @code_warntype checked 2021/11/19
+(p::Point_2D, n::Real) = Point_2D(p.x[1] + n, p.x[2] + n)
# @code_warntype checked 2021/11/19
+(n::Real, p::Point_2D) = p + n
# @code_warntype checked 2021/11/19
-(p::Point_2D, n::Real) = Point_2D(p.x[1] - n, p.x[2] - n)
# @code_warntype checked 2021/11/19
-(n::Real, p::Point_2D) = p - n
# @code_warntype checked 2021/11/19
*(n::Real, p::Point_2D) = Point_2D(n*p.x[1], n*p.x[2])
# @code_warntype checked 2021/11/19
*(p::Point_2D, n::Real) = n*p
# @code_warntype checked 2021/11/19
/(p::Point_2D, n::Real) = Point_2D(p.x[1]/n, p.x[2]/n)
# @code_warntype checked 2021/11/19
-(p::Point_2D) = -1*p
# SMatrix multiplication, returns a point
# @code_warntype checked 2021/11/19
*(A::SMatrix{2, 2, F, 4}, p::Point_2D{F}) where {F <: AbstractFloat} = Point_2D(A * p.x)

# Methods
# -------------------------------------------------------------------------------------------------
# note: hypot is the Julia recommended way to do sqrt of sum squared for 2 numbers
# @code_warntype checked 2021/11/19
norm(p::Point_2D) = hypot(p.x[1], p.x[2])
# @code_warntype checked 2021/11/19
function distance(p₁::Point_2D, p₂::Point_2D)
    return hypot(p₁.x[1] - p₂.x[1], p₁.x[2] - p₂.x[2])
end
# @code_warntype checked 2021/11/19
midpoint(p₁::Point_2D, p₂::Point_2D) = (p₁ + p₂)/2

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    convert_arguments(S::Type{<:Scatter}, p::Point_2D) = convert_arguments(S, p.x)
    function convert_arguments(S::Type{<:Scatter}, P::Vector{<:Point_2D})
        return convert_arguments(S, [p.x for p in P])
    end
end
