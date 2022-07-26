import LinearAlgebra: cross

struct Vec2{T}
    x::T
    y::T
end

function Base.show(io::IO, v::Vec2)
    return print(io, "(", v.x, ", ", v.y, ")")
end

Base.:+(v1::Vec2, v2::Vec2) = Vec2(v1.x + v2.x, v1.y + v2.y)
cross(v1::Vec2, v2::Vec2) = v1.x * v2.y - v1.y * v2.x
cross(v1::Vector, v2::Vector) = v1[1] * v2[2] - v1[2] * v2[1]
