# A 3D point in Cartesian coordinates.

struct Point_3D{T <: AbstractFloat}
    coord::SVector{3,T}
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
                                              X,Y,Z <: Real} = Point_3D(SVector(T(x),T(y),T(z)))
# 2D single conversion constructor
Point_3D(type::Type{T}, x::X, y::Y) where {T <: AbstractFloat,
                                       X,Y <: Real} = Point_3D(SVector(T(x),T(y),T(0)))
# 1D single conversion constructor
Point_3D(type::Type{T}, x::X) where {T <: AbstractFloat,
                                 X <: Real} = Point_3D(SVector(T(x),T(0),T(0)))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(p⃗::Point_3D) = Ref(p⃗)
Base.zero(::Point_3D{T}) where {T <: AbstractFloat} = Point_3D((T(0), T(0), T(0)))
Base.firstindex(::Point_3D) = 1
Base.lastindex(::Point_3D) = 3
Base.getindex(p⃗::Point_3D, i::Int) = p⃗.coord[i]
(::Type{T})(p⃗::Point_3D) where {T <: AbstractFloat} = Point_3D(T.(p⃗.coord))

# Operators
# -------------------------------------------------------------------------------------------------
==(p⃗₁::Point_3D, p⃗₂::Point_3D) = (p⃗₁.coord == p⃗₂.coord)
function ≈(p⃗₁::Point_3D{T}, p⃗₂::Point_3D{T}) where {T <: AbstractFloat}
    return distance(p⃗₁, p⃗₂) < 5.0e-5 
end
+(p⃗₁::Point_3D, p⃗₂::Point_3D) = Point_3D(p⃗₁.coord[1] + p⃗₂.coord[1],
                                         p⃗₁.coord[2] + p⃗₂.coord[2],
                                         p⃗₁.coord[3] + p⃗₂.coord[3]
                                        )
-(p⃗₁::Point_3D, p⃗₂::Point_3D) = Point_3D(p⃗₁.coord[1] - p⃗₂.coord[1],
                                         p⃗₁.coord[2] - p⃗₂.coord[2],
                                         p⃗₁.coord[3] - p⃗₂.coord[3]
                                        )
×(p⃗₁::Point_3D, p⃗₂::Point_3D) = Point_3D(p⃗₁.coord[2]*p⃗₂.coord[3] - p⃗₂.coord[2]*p⃗₁.coord[3],
                                         p⃗₁.coord[3]*p⃗₂.coord[1] - p⃗₂.coord[3]*p⃗₁.coord[1],
                                         p⃗₁.coord[1]*p⃗₂.coord[2] - p⃗₂.coord[1]*p⃗₁.coord[2],
                                )
⋅(p⃗₁::Point_3D, p⃗₂::Point_3D) = p⃗₁.coord[1]*p⃗₂.coord[1] + 
                                p⃗₁.coord[2]*p⃗₂.coord[2] + 
                                p⃗₁.coord[3]*p⃗₂.coord[3]

+(p⃗::Point_3D, n::Real) = Point_3D(p⃗.coord[1] + n,
                                   p⃗.coord[2] + n,
                                   p⃗.coord[3] + n
                                  )
+(n::Real,  p⃗::Point_3D) = p⃗ + n
-(p⃗::Point_3D, n::Real) = Point_3D(p⃗.coord[1] - n,
                                   p⃗.coord[2] - n,
                                   p⃗.coord[3] - n
                                  )
-(n::Real,  p⃗::Point_3D) = p⃗ - n
*(n::Real,  p⃗::Point_3D) = Point_3D(p⃗.coord[1] * n,
                                    p⃗.coord[2] * n,
                                    p⃗.coord[3] * n
                                   )
*(p⃗::Point_3D, n::Real) = n*p⃗
/(p⃗::Point_3D, n::Real) = Point_3D(p⃗.coord[1] / n,
                                   p⃗.coord[2] / n,
                                   p⃗.coord[3] / n
                                  )
-(p⃗::Point_3D) = -1*p⃗

# Methods
# -------------------------------------------------------------------------------------------------
norm(p⃗::Point_3D) = sqrt(p⃗ ⋅ p⃗)
distance(p⃗₁::Point_3D, p⃗₂::Point_3D) = norm(p⃗₁ - p⃗₂)

# Plot
# -------------------------------------------------------------------------------------------------
convert_arguments(P::Type{<:Scatter}, p::Point_3D) = convert_arguments(P, p.coord)
function convert_arguments(P::Type{<:Scatter}, AP::AbstractArray{<:Point_3D})
    return convert_arguments(P, [p.coord for p in AP])
end
