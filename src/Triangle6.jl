import Base: intersect

struct Triangle6{T <: AbstractFloat} <: Face
    points::NTuple{6, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle6(p₁::Point{T}, 
         p₂::Point{T}, 
         p₃::Point{T},
         p₄::Point{T},
         p₅::Point{T},
         p₆::Point{T}
        ) where {T <: AbstractFloat} = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))


# Methods
# -------------------------------------------------------------------------------------------------
function (tri6::Triangle6)(r::T, s::T) where {T <: AbstractFloat}
    w₁ = (1 - r - s)*(2(1 - r - s) - 1)
    w₂ = r*(2r-1)
    w₃ = s*(2s-1)
    w₄ = 4r*(1 - r - s)
    w₅ = 4r*s
    w₆ = 4s*(1 - r - s)
    weights = [w₁, w₂, w₃, w₄, w₅, w₆]
    return sum(weights .* tri6.points) 
end

# eval, 
# area, integrate or trianglulate.

# triangulate.
#function intersect(l::LineSegment, tri::Triangle)
#end
