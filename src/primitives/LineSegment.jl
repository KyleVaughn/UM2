# A line segment, defined as the set of all points such that 𝗹(r) = 𝘅₁ + r𝘂, 
# where r ∈ [0, 1]. 𝘅₁ is the line segment start and 𝘅₂ = 𝘅₁ + 𝘂 is the line 
# segment end.
#
# We store 𝘂 instead of 𝘅₂, since 𝘅₂ is needed infrequently, but 𝘂 is needed often.
struct LineSegment{Dim, T} <:Edge{Dim, 1, T}
    𝘅₁::Point{Dim, T} 
    𝘂::SVector{Dim, T}
end

const LineSegment1D = LineSegment{1}
const LineSegment2D = LineSegment{2}
const LineSegment3D = LineSegment{3}

function Base.getproperty(l::LineSegment, sym::Symbol)
    if sym === :𝘅₂
        return Point(l.𝘅₁ + l.𝘂)
    else # fallback to getfield
        return getfield(l, sym)
    end
end

# Constructors
# ---------------------------------------------------------------------------------------------
# Points
LineSegment{Dim, T}(𝘅₁::Point{Dim, T}, 
                    𝘅₂::Point{Dim, T}) where {Dim, T} = LineSegment{Dim, T}(𝘅₁, 𝘅₂ - 𝘅₁) 
LineSegment{Dim}(𝘅₁::Point{Dim, T}, 
                 𝘅₂::Point{Dim, T}) where {Dim, T} = LineSegment{Dim, T}(𝘅₁, 𝘅₂ - 𝘅₁) 
LineSegment(𝘅₁::Point{Dim, T}, 
            𝘅₂::Point{Dim, T}) where {Dim, T} = LineSegment{Dim, T}(𝘅₁, 𝘅₂ - 𝘅₁) 
# Vector
LineSegment{Dim, T}(pts::SVector{2, Point{Dim, T}}
                   ) where {Dim, T} = LineSegment{Dim, T}(pts[1], pts[2] - pts[1]) 
LineSegment{Dim}(pts::SVector{2, Point{Dim, T}}
                ) where {Dim, T} = LineSegment{Dim, T}(pts[1], pts[2] - pts[1]) 
LineSegment(pts::SVector{2, Point{Dim, T}}
           ) where {Dim, T} = LineSegment{Dim, T}(pts[1], pts[2] - pts[1]) 

# Methods
# ---------------------------------------------------------------------------------------------
# Interpolation
# Note: 𝗹(0) = 𝘅₁, 𝗹(1) = 𝘅₂
@inline (l::LineSegment)(r) = Point(l.𝘅₁.coord + r*l.𝘂)
@inline arclength(l::LineSegment) = distance(l.𝘅₁.coord, l.𝘅₁.coord + l.𝘂)
function Base.intersect(l₁::LineSegment1D{T}, l₂::LineSegment1D{T}) where {T}
    l₁_start = l₁.𝘅₁[1]
    l₁_stop  = l₁_start + l₁.𝘂[1]
    l₂_start = l₂.𝘅₁[1]
    l₂_stop  = l₂_start + l₂.𝘂[1]
    min₁ = min(l₁_start, l₁_stop) 
    max₁ = max(l₁_start, l₁_stop) 
    min₂ = min(l₂_start, l₂_stop) 
    max₂ = max(l₂_start, l₂_stop) 
    if min₂ ≤ min₁ ≤ max₂     # (₂--(₁------₂) 
        if max₁ ≤ max₂        # (₂--(₁--₁)--₂) 
            return (true, l₁)
        else                  # (₂--(₁------₂)--₁) 
            return (true, LineSegment1D(Point(min₁), Point(max₂)))
        end
    elseif min₂ ≤ max₁ ≤ max₂ # (₁--(₂------₁)--₂)  
        return (true, LineSegment1D(Point(min₂), Point(max₁)))
    elseif min₁ ≤ min₂ && max₂ ≤ max₁ # (₁--(₂--₂)--₁) 
        return (true, l₂)
    else # disjoint
        return (false, l₁)
    end
end
# Intersection of two 2D or 3D line segments
#
# Doesn't work for colinear/parallel lines. (𝘂 × 𝘃 = 𝟬).
# For 𝗹₁(r) = 𝘅₁ + r𝘂 and 𝗹₂(s) = 𝘅₂ + s𝘃
# 1) 𝘅₁ + r𝘂 = 𝘅₂ + s𝘃                  subtracting 𝘅₁ from both sides
# 2) r𝘂 = (𝘅₂-𝘅₁) + s𝘃                  𝘄 = 𝘅₂-𝘅₁
# 3) r𝘂 = 𝘄 + s𝘃                        cross product with 𝘃 (distributive)
# 4) r(𝘂 × 𝘃) = 𝘄 × 𝘃 + s(𝘃 × 𝘃)        𝘃 × 𝘃 = 𝟬
# 5) r(𝘂 × 𝘃) = 𝘄 × 𝘃                   let 𝘄 × 𝘃 = 𝘅 and 𝘂 × 𝘃 = 𝘇
# 6) r𝘇 = 𝘅                             dot product 𝘇 to each side
# 7) r𝘇 ⋅ 𝘇 = 𝘅 ⋅ 𝘇                     divide by 𝘇 ⋅ 𝘇
# 8) r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)
# We need to ensure r, s ∈ [0, 1], hence we need to solve for s too.
# 1) 𝘅₂ + s𝘃 = 𝘅₁ + r𝘂                     subtracting 𝘅₂ from both sides
# 2) s𝘃 = -𝘄 + r𝘂                          cross product with 𝘄
# 3) s(𝘃 × 𝘄) = -𝘄 × 𝘄 + r(𝘂 × 𝘄)          𝘄 × 𝘄 = 𝟬 
# 4) s(𝘃 × 𝘄) = r(𝘂 × 𝘄)                   using 𝘂 × 𝘄 = -(𝘄 × 𝘂), likewise for 𝘃 × 𝘄
# 5) s(𝘄 × 𝘃) = r(𝘄 × 𝘂)                   let 𝘄 × 𝘂 = 𝘆. use 𝘄 × 𝘃 = 𝘅
# 6) s𝘅 = r𝘆                               dot product 𝘅 to each side
# 7) s(𝘅 ⋅ 𝘅) = r(𝘆 ⋅ 𝘅)                   divide by (𝘅 ⋅ 𝘅)
# 9) s = r(𝘅 ⋅ 𝘆)/(𝘅 ⋅ 𝘅)
# The cross product of two vectors in the plane is a vector of the form (0, 0, k),
# hence, in 2D:
# r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇) = x₃/z₃ 
# s = r(𝘅 ⋅ 𝘆)/(𝘅 ⋅ 𝘅) = y₃/z₃ 
function Base.intersect(l₁::LineSegment2D{T}, l₂::LineSegment2D{T}) where {T}
    ϵ = T(5e-6) # Tolerance on r,s ∈ [-ϵ, 1 + ϵ]
    𝘄 = l₂.𝘅₁ - l₁.𝘅₁
    z = l₁.𝘂 × l₂.𝘂
    r = (𝘄 × l₂.𝘂)/z
    s = (𝘄 × l₁.𝘂)/z
    return (T(1e-8) < abs(z) && -ϵ ≤ r ≤ 1 + ϵ 
                             && -ϵ ≤ s ≤ 1 + ϵ, l₂(s)) # (hit, point)
end
function Base.intersect(l₁::LineSegment3D{T}, l₂::LineSegment3D{T}) where {T}
    ϵ = T(5e-6) # Tolerance on r,s ∈ [-ϵ, 1 + ϵ]
    𝘂 = l₁.𝘂
    𝘃 = l₂.𝘂
    𝘇 = 𝘂 × 𝘃
    𝘄 = l₂.𝘅₁ - l₁.𝘅₁
    𝘅 = 𝘄 × 𝘃 
    𝘆 = 𝘄 × 𝘂
    r = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)
    s = r*(𝘅 ⋅ 𝘆)/(𝘅 ⋅ 𝘅)
    return (T(1e-16) < norm²(𝘇) && -ϵ ≤ r ≤ 1 + ϵ 
                                && -ϵ ≤ s ≤ 1 + ϵ, l₂(s)) # (hit, point)
end

# If the point is left of the line segment in the 2D plane. 
#
# The segment's direction is from 𝘅₁ to 𝘅₂. Let 𝘂 = 𝘅₂ - 𝘅₁ and 𝘃 = 𝗽 - 𝘅₁ 
# We may determine if the angle θ between the point and segment is in [0, π] based on the 
# sign of 𝘂 × 𝘃, since 𝘂 × 𝘃 = ‖𝘂‖‖𝘃‖sin(θ). 
#   𝗽    ^
#   ^   /
# 𝘃 |  / 𝘂
#   | /
#   o
# We allow points on the line (𝘂 × 𝘃 = 0) to be left, since this test is primarily 
# used to determine if a point is inside a polygon. A mesh is supposed to partition
# its domain, so if we do not allow points on the line, there will exist points in the 
# mesh which will not be in any face, violating that rule.
@inline function isleft(p::Point2D, l::LineSegment2D)
    return 0 ≤ l.𝘂 × (p - l.𝘅₁) 
end

# Random line in the Dim-dimensional unit hypercube
function Base.rand(::Type{LineSegment{Dim, F}}) where {Dim, F} 
    points = rand(Point{Dim, F}, 2)
    return LineSegment{Dim, F}(points[1], points[2])
end

# N random lines in the Dim-dimensional unit hypercube
function Base.rand(::Type{LineSegment{Dim, F}}, N::Int64) where {Dim, F}
    return [ rand(LineSegment{Dim, F}) for i ∈ 1:N ]
end

# Sort intersection points along a line segment, deleting points that are less than 
# the minimum_segment_length apart
function sort_intersection_points!(l::LineSegment, points::Vector{<:Point})
    sort!(l.𝘅₁, points)
    id_start = 1 
    n = length(points)
    deletion_indices = Int64[]
    for id_stop ∈ 2:n
        if distance²(points[id_start], points[id_stop]) < minimum_segment_length^2
            push!(deletion_indices, id_stop)
        else
            id_start = id_stop
        end
    end
    deleteat!(points, deletion_indices)
    return points
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
        return convert_arguments(LS, [l.𝘅₁, l.𝘅₂])
    end

    function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment})
        return convert_arguments(LS, reduce(vcat, [[l.𝘅₁, l.𝘅₂] for l in L]))
    end
end
