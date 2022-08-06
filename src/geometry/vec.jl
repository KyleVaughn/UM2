export Vec,
       Vec2,  Vec3,
       Vec2f, Vec3f,
       Vec2d, Vec3d,
       Vec2b, Vec3b


# -- Type aliases --

const Vec = SVector
const Vec2  = Vec{2}
const Vec3  = Vec{3}
const Vec2f = Vec{2, Float32}
const Vec3f = Vec{3, Float32}
const Vec2d = Vec{2, Float64}
const Vec3d = Vec{3, Float64}
const Vec2b = Vec{2, BigFloat}
const Vec3b = Vec{3, BigFloat}

# -- Conversion --

function Base.convert(::Type{Vec{2, T}}, v::Vec{3}) where {T}
    return Vec{2, T}(v.coord[1], v.coord[2])
end
