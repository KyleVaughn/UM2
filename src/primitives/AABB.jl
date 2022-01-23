# Axis-aligned bounding box.
# An Dim-dimensional box requires 2 Dim-dimensional points to specify the boundary:
#   One point to specify the box origin, and one to specify the opposite (furthest corner)
struct AABB{Dim,T}
    origin::Point{Dim,T}
    corner::Point{Dim,T}
end

const AABB2D = AABB{2}
const AABB3D = AABB{3}

# Dimote: all branches but the correct one are pruned by the compiler
function Base.getproperty(aabb::AABB, sym::Symbol)
    if sym === :xmin
        return aabb.origin[1]
    elseif sym === :ymin
        return aabb.origin[2]
    elseif sym === :zmin
        return aabb.origin[3]
    elseif sym === :xmax
        return aabb.corner[1]
    elseif sym === :ymax
        return aabb.corner[2]
    elseif sym === :zmax
        return aabb.corner[3]
    else # fallback to getfield
        return getfield(aabb, sym)
    end
end

# Constructors
# ---------------------------------------------------------------------------------------------
AABB{Dim}(p₁::Point{Dim,T}, p₂::Point{Dim,T}) where {Dim,T} = AABB{Dim,T}(p₁, p₂)

# Methods
# -------------------------------------------------------------------------------------------------
@inline width(aabb::AABB) = aabb.xmax - aabb.xmin
@inline height(aabb::AABB) = aabb.ymax - aabb.ymin
@inline depth(aabb::AABB) = aabb.ymax - aabb.ymin
@inline area(aabb::AABB2D) = height(aabb) * width(aabb)
@inline volume(aabb::AABB3D) = height(aabb) * width(aabb) * depth(aabb)
@inline Base.in(p::Point2D, aabb::AABB2D) = aabb.xmin ≤ p[1] ≤ aabb.xmax && 
                                            aabb.ymin ≤ p[2] ≤ aabb.ymax
@inline Base.in(p::Point3D, aabb::AABB3D) = aabb.xmin ≤ p[1] ≤ aabb.xmax && 
                                            aabb.ymin ≤ p[2] ≤ aabb.ymax &&
                                            aabb.zmin ≤ p[3] ≤ aabb.zmax

# DEPRECATED. Leaving because there is potential future use
# # Liang-Barsky line clipping algorithm
# # pₖ = 0	            parallel to the clipping boundaries
# # pₖ = 0 and qₖ < 0	    completely outside the boundary
# # pₖ = 0 and qₖ ≥ 0	    inside the parallel clipping boundary
# # pₖ < 0	            line proceeds from outside to inside
# # pₖ > 0	            line proceeds from inside to outside
# function intersect(l::LineSegment2D{F}, aabb::AABB2D{F}) where {F <: AbstractFloat}
#     p₂ = l[2].x - l[1].x
#     p₁ = -p₂
#     p₄ = l[2].y - l[1].y
#     p₃ = -p₄
# 
#     q₁ = l[1].x - aabb.xmin
#     q₂ = aabb.xmax - l[1].x
#     q₃ = l[1].y - aabb.ymin
#     q₄ = aabb.ymax - l[1].y
# 
#     # Line parallel to clipping window
#     if p₁ == 0 # Vertical line
#         if q₁ < 0 || q₂ < 0 # Outside boundaries
#             return false, SVector(Point2D{F}(0, 0), Point2D{F}(0, 0))
#         else # Inside boundaries
#             return true, SVector(Point2D(l[1].x, aabb.ymin), Point2D(l[1].x, aabb.ymax))
#         end
#     end
#     if p₃ == 0 # Horizontal line
#         if q₃ < 0 || q₄ < 0 # Outside boundaries
#             return false, SVector(Point2D{F}(0, 0), Point2D{F}(0, 0))
#         else # Inside boundaries
#             return true, SVector(Point2D(aabb.xmin, l[1].y), Point2D(aabb.xmax, l[1].y))
#         end
#     end
# 
#     t₁ = q₁ / p₁
#     t₂ = q₂ / p₂
#     if (p₁ < 0)
#         t_min2 = t₁
#         t_max2 = t₂
#     else
#         t_min2 = t₂
#         t_max2 = t₁
#     end
# 
#     t₃ = q₃ / p₃
#     t₄ = q₄ / p₄
#     if (p₃ < 0)
#         t_min3 = t₃
#         t_max3 = t₄
#     else
#         t_min3 = t₄
#         t_max3 = t₃
#     end
# 
#     t_start = max(F(0), t_min2, t_min3)
#     t_stop = min(F(1), t_max2, t_max3)
# 
#     # Line outside clipping window
#     t_start < t_stop || return false, SVector(Point2D{F}(0, 0), Point2D{F}(0, 0))
# 
#     return true, SVector(l(t_start), l(t_stop))
# end
# 
# Credit to Tavian Barnes (https://tavianator.com/2011/ray_box.html)
# Assumes the line passes all the way through the AABB if it intersects, which is a 
# valid assumption for this ray tracing application. 
function Base.intersect(l::LineSegment2D, aabb::AABB2D)
    𝘁₁ = (aabb.origin - l.𝘅₁) ./ l.𝘂
    𝘁₂ = (aabb.corner - l.𝘅₁) ./ l.𝘂

    tmin = max(min(𝘁₁[1], 𝘁₂[1]), min(𝘁₁[2], 𝘁₂[2]))
    tmax = min(max(𝘁₁[1], 𝘁₂[1]), max(𝘁₁[2], 𝘁₂[2]))

    return (tmax >= tmin, SVector(l(tmin), l(tmax)))
end

# A random AABB within [0, 1]ᴺ ⊂ ℝᴺ
# What does the distribution of AABBs look like? Is this uniform? 
function Base.rand(::Type{AABB{Dim,T}}) where {Dim,T}
    coord₁ = rand(T, Dim)
    coord₂ = rand(T, Dim)
    return AABB{Dim,T}(Point{Dim,T}(min.(coord₁, coord₂)), 
                       Point{Dim,T}(max.(coord₁, coord₂)))  
end

# N random AABB within [0, 1]ᴺ ⊂ ℝᴺ
function Base.rand(::Type{AABB{Dim,T}}, N::Int64) where {Dim,T}
    return [ rand(AABB{Dim,T}) for i ∈ 1:N ]
end

# Return the AABB which contains both bb₁ and bb₂
function Base.union(bb₁::AABB{Dim,T}, bb₂::AABB{Dim,T}) where {Dim,T}
    return AABB(Point{Dim,T}(min.(bb₁.origin.coord, bb₂.origin.coord)),
                Point{Dim,T}(max.(bb₁.corner.coord, bb₂.corner.coord)))
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, aabb::AABB2D)
        p₂ = Point2D(aabb.xmax, aabb.ymin)
        p₄ = Point2D(aabb.xmin, aabb.ymax)
        l₁ = LineSegment2D(aabb.origin, p₂)
        l₂ = LineSegment2D(p₂, aabb.corner)
        l₃ = LineSegment2D(aabb.corner, p₄)
        l₄ = LineSegment2D(p₄, aabb.origin)
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABB2D})
        point_sets = [convert_arguments(LS, aabb) for aabb in R]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, aabb::AABB2D)
        p₂ = Point2D(aabb.xmax, aabb.ymin)
        p₄ = Point2D(aabb.xmin, aabb.ymax)
        points = [aabb.origin.coord, p₂.coord, aabb.corner.coord, p₄.coord]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, R::Vector{<:AABB2D})
        points = reduce(vcat, [[aabb.origin.coord, 
                                Point2D(aabb.xmax, aabb.ymin).coord,
                                aabb.corner.coord, 
                                Point2D(aabb.xmin, aabb.ymax).coord] for aabb ∈ R])
        faces = zeros(Int64, 2*length(R), 3)
        j = 0
        for i in 1:2:2*length(R)
            faces[i    , :] = [1 2 3] .+ j
            faces[i + 1, :] = [3 4 1] .+ j
            j += 4
        end
        return convert_arguments(M, points, faces)
    end
end
