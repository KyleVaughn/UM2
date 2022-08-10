export Ray

struct Ray{D, T <: AbstractFloat}
    origin::Point{D, T}
    direction::Vec{D, T}
end

# -- Type aliases --

const Ray2f = Ray{2, Float32}
const Ray2d = Ray{2, Float64}
const Ray2b = Ray{2, BigFloat}
