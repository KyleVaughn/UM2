export Vec,
       Vec1, Vec2, Vec3,
       Vec1f, Vec2f, Vec3f,
       Vec1d, Vec2d, Vec3d

export ⋅, ×, coord, dot, cross, norm2, norm, normalize

# VECTOR
# --------------------------------------------------------------------------- 
#
# A D-dimenstional vector with data of type T
#

struct Vec{D, T <: AbstractFloat} <: AbstractVector{T}
   coord::NTuple{D, T}
end

# -- Type aliases --

const Vec1  = Vec{1}
const Vec2  = Vec{2}
const Vec3  = Vec{3}

const Vec1f = Vec1{f32}
const Vec2f = Vec2{f32}
const Vec3f = Vec3{f32}

const Vec1d = Vec1{f64}
const Vec2d = Vec2{f64}
const Vec3d = Vec3{f64}

# -- Abstract vector interface --

Base.getindex(v::Vec, i::Integer) = Base.getindex(v.coord, i)
Base.size(v::Vec{D}) where {D} = (D,)
Base.length(v::Vec{D}) where {D} = D 

# -- Accessors --

coord(v::Vec) = v.coord

# -- Constructors --

Vec(xs::T...) where {T} = Vec{length(xs), T}(xs)
Vec{D}(xs::T...) where {D, T} = Vec{D, T}(xs)
Vec{D, T}(xs::X...) where {D, T, X <: Number} = Vec{D, T}(map(T, xs))

# -- Unary operators --

Base.:-(v::Vec) = Vec(map(-, v.coord))

# -- Binary operators --

Base.:+(v::Vec{D, T}, scalar::X) where {D, T, X <: Number} = Vec(map(x -> x + T(scalar), v.coord))
Base.:-(v::Vec{D, T}, scalar::X) where {D, T, X <: Number} = Vec(map(x -> x - T(scalar), v.coord)) 
Base.:*(v::Vec{D, T}, scalar::X) where {D, T, X <: Number} = Vec(map(x -> x * T(scalar), v.coord)) 
Base.:/(v::Vec{D, T}, scalar::X) where {D, T, X <: Number} = Vec(map(x -> x / T(scalar), v.coord)) 
Base.:+(scalar::X, v::Vec{D, T}) where {D, T, X <: Number} = Vec(map(x -> T(scalar) + x, v.coord)) 
Base.:-(scalar::X, v::Vec{D, T}) where {D, T, X <: Number} = Vec(map(x -> T(scalar) - x, v.coord)) 
Base.:*(scalar::X, v::Vec{D, T}) where {D, T, X <: Number} = Vec(map(x -> T(scalar) * x, v.coord)) 
Base.:/(scalar::X, v::Vec{D, T}) where {D, T, X <: Number} = Vec(map(x -> T(scalar) / x, v.coord)) 
Base.:+(u::Vec{D, T}, v::Vec{D, T}) where {D, T} = Vec(map(+, u.coord, v.coord))
Base.:-(u::Vec{D, T}, v::Vec{D, T}) where {D, T} = Vec(map(-, u.coord, v.coord))
Base.:*(u::Vec{D, T}, v::Vec{D, T}) where {D, T} = Vec(map(*, u.coord, v.coord))
Base.:/(u::Vec{D, T}, v::Vec{D, T}) where {D, T} = Vec(map(/, u.coord, v.coord))

# -- Methods --

dot(u::Vec{D, T}, v::Vec{D, T}) where {D, T} = mapreduce(*, +, u.coord, v.coord)
const ⋅ = dot
norm2(v::Vec) = dot(v, v)
norm(v::Vec) = sqrt(norm2(v))
normalize(v::Vec) = v / norm(v)

# -- 2D vector methods --

cross(u::Vec2, v::Vec2) = u[1] * v[2] - u[2] * v[1]
const × = cross
