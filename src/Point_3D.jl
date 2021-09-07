# A 3D point in Cartesian coordinates.

struct Point_3D{T <: AbstractFloat}
    x::SVector{3,T}
end

# Constructors
# -------------------------------------------------------------------------------------------------
# 3D single constructor
Point_3D(x::T, y::T, z::T) where {T <: AbstractFloat} = Point_3D(SVector(x,y,z))
# 2D single constructor
Point_3D(x::T, y::T) where {T <: AbstractFloat} = Point_3D(SVector(x, y, T(0)))
# 1D single constructor
Point_3D(x::T) where {T <: AbstractFloat} = Point_3D(SVector(x, T(0), T(0)))
# 3D tuple constructor
Point_3D(x::Tuple{T,T,T}) where {T <: AbstractFloat} = Point_3D(SVector(x))
# 2D tuple constructor
Point_3D((x, y)::Tuple{T,T}) where {T <: AbstractFloat} = Point_3D(SVector(x, y, T(0)))
# 3D single conversion constructor
Point_3D(type::Type{T}, x::X, y::Y, z::Z) where {T <: AbstractFloat,
                                                 X <: Real,
                                                 Y <: Real,
                                                 Z <: Real} = Point_3D(SVector(T(x),T(y),T(z)))
# 2D single conversion constructor
Point_3D(type::Type{T}, x::X, y::Y) where {T <: AbstractFloat,
                                           X <: Real,
                                           Y <: Real} = Point_3D(SVector(T(x),T(y),T(0)))
# 1D single conversion constructor
Point_3D(type::Type{T}, x::X) where {T <: AbstractFloat,
                                     X <: Real} = Point_3D(SVector(T(x),T(0),T(0)))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p⃗::Point_3D) = Ref(p⃗)
Base.zero(::Point_3D{T}) where {T <: AbstractFloat} = Point_3D((T(0), T(0), T(0)))
Base.firstindex(::Point_3D) = 1
Base.lastindex(::Point_3D) = 3
Base.getindex(p⃗::Point_3D, i::Int) = p⃗.x[i]
(::Type{T})(p⃗::Point_3D) where {T <: AbstractFloat} = Point_3D(T.(p⃗.x))

# Operators
# -------------------------------------------------------------------------------------------------
==(p⃗₁::Point_3D, p⃗₂::Point_3D) = (p⃗₁.x == p⃗₂.x)
function ≈(p⃗₁::Point_3D{T}, p⃗₂::Point_3D{T}) where {T <: AbstractFloat}
    return distance(p⃗₁, p⃗₂) < 5e-6 
end
+(p⃗₁::Point_3D, p⃗₂::Point_3D) = Point_3D(p⃗₁.x[1] + p⃗₂.x[1],
                                         p⃗₁.x[2] + p⃗₂.x[2],
                                         p⃗₁.x[3] + p⃗₂.x[3]
                                        )
-(p⃗₁::Point_3D, p⃗₂::Point_3D) = Point_3D(p⃗₁.x[1] - p⃗₂.x[1],
                                         p⃗₁.x[2] - p⃗₂.x[2],
                                         p⃗₁.x[3] - p⃗₂.x[3]
                                        )
×(p⃗₁::Point_3D, p⃗₂::Point_3D) = Point_3D(p⃗₁.x[2]*p⃗₂.x[3] - p⃗₂.x[2]*p⃗₁.x[3],
                                         p⃗₁.x[3]*p⃗₂.x[1] - p⃗₂.x[3]*p⃗₁.x[1],
                                         p⃗₁.x[1]*p⃗₂.x[2] - p⃗₂.x[1]*p⃗₁.x[2],
                                )

⋅(p⃗₁::Point_3D, p⃗₂::Point_3D) = p⃗₁.x[1]*p⃗₂.x[1] + 
                                p⃗₁.x[2]*p⃗₂.x[2] + 
                                p⃗₁.x[3]*p⃗₂.x[3]

+(p⃗::Point_3D, n::Real) = Point_3D(p⃗.x[1] + n,
                                   p⃗.x[2] + n,
                                   p⃗.x[3] + n
                                  )
+(n::Real,  p⃗::Point_3D) = p⃗ + n
-(p⃗::Point_3D, n::Real) = Point_3D(p⃗.x[1] - n,
                                   p⃗.x[2] - n,
                                   p⃗.x[3] - n
                                  )
-(n::Real,  p⃗::Point_3D) = p⃗ - n
*(n::Real,  p⃗::Point_3D) = Point_3D(p⃗.x[1] * n,
                                    p⃗.x[2] * n,
                                    p⃗.x[3] * n
                                   )
*(p⃗::Point_3D, n::Real) = n*p⃗
/(p⃗::Point_3D, n::Real) = Point_3D(p⃗.x[1] / n,
                                   p⃗.x[2] / n,
                                   p⃗.x[3] / n
                                  )
-(p⃗::Point_3D) = -1*p⃗
# 3 x 3 matrix multiplication, returns a point
*(A::SMatrix{3, 3, T, 9}, p⃗::Point_3D{T}) where {T <: AbstractFloat} = Point_3D(A * p⃗.x)

# Methods
# -------------------------------------------------------------------------------------------------
norm(p⃗::Point_3D) = sqrt(p⃗ ⋅ p⃗)
distance(p⃗₁::Point_3D, p⃗₂::Point_3D) = norm(p⃗₁ - p⃗₂)
