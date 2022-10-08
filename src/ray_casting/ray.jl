export Ray,
       Ray2,
       Ray2f,
       Ray2d

export interpolate_ray

# RAY 
# -----------------------------------------------------------------------------
#
# A ray represented by its origin and a displacement vector.
# Note the vector is not normalized.
#

struct Ray{D, T} 
    origin::Point{D, T}
    direction::Vec{D, T} # Normalized
end

# -- Type aliases --

const Ray2  = Ray{2}
const Ray2f = Ray2{Float32}
const Ray2d = Ray2{Float64}

# -- Interpolation --

function interpolate_ray(origin::T, direction::T, r) where {T}
    return origin + r * direction
end

function (ray::Ray{D, T})(r::T) where {D, T}
    return ray.origin + r * ray.direction
end

# -- IO --

function Base.show(io::IO, r::Ray{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "Ray", D, type_char, '(',
        r.origin, ", ",
        r.direction, ')')
end
