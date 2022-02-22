# Specialized methods for a Quadrilateral, aka Polygon{4}
(quad::Quadrilateral)(r, s) = Point(((1 - r)*(1 - s))quad[1] + (r*(1 - s))quad[2] + 
                                                (r*s)quad[3] + ((1 - r)*s)quad[4])
# This performs much better than the default routine, which is logically equivalent.
# Better simd this way? Chaining isleft doesn't have the same performance improvement for
# triangles.
function Base.in(p::Point2D, quad::Quadrilateral2D)
    return isleft(p, LineSegment2D(quad[1], quad[2])) &&
           isleft(p, LineSegment2D(quad[2], quad[3])) &&
           isleft(p, LineSegment2D(quad[3], quad[4])) &&
           isleft(p, LineSegment2D(quad[4], quad[1]))
end

# Intersect
# ---------------------------------------------------------------------------------------------
function intersect(l::LineSegment2D{T}, quad::Quadrilateral2D{T}
                  ) where {T <: Union{Float32, Float64}} 
    hit₁, p₁ = l ∩ LineSegment2D(quad[1], quad[2])
    hit₂, p₂ = l ∩ LineSegment2D(quad[2], quad[3])
    hit₃, p₃ = l ∩ LineSegment2D(quad[3], quad[4])
    hit₄, p₄ = l ∩ LineSegment2D(quad[4], quad[1])
    # Possibilities: 1+2, 1+4, 2+3, 2+4,1+2+3+4, none. 
    # Only return 3 points, since this will return all unique points
    if hit₁
        if hit₂
            if hit₃ # 1+2+3
                return 0x0003, SVector(p₁, p₂, p₃)
            else # 1+2
                return 0x0002, SVector(p₁, p₂, p₃)
            end
        else # 1+4
            return 0x0002, SVector(p₁, p₄, p₂)
        end
    elseif hit₂ # 2+3
        return 0x0002, SVector(p₂, p₃, p₁)
    else # none
        return 0x0000, SVector(p₁, p₂, p₃)
    end
end
