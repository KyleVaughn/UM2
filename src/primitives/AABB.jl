# Axis-aligned bounding box.
# An N-dimensional box requires 2 N-dimensional points to specify the boundary:
#   One point to specify the box origin, and one to specify the opposite (furthest corner)
struct AABB{N,T} <: Face{N,T}
    origin::Point{N,T}
    corner::Point{N,T}
end

const AABB_2D = AABB{2}
const AABB_3D = AABB{3}

# Note: all branches but the correct one are pruned by the compiler
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
AABB{N}(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = AABB{N,T}(p₁, p₂)
AABB(p₁::Point{N,T}, p₂::Point{N,T}) where {N,T} = AABB{N,T}(p₁, p₂)

# Methods
# -------------------------------------------------------------------------------------------------
@inline width(aabb::AABB) = aabb.xmax - aabb.xmin
@inline height(aabb::AABB) = aabb.ymax - aabb.ymin
@inline depth(aabb::AABB) = aabb.ymax - aabb.ymin
@inline area(aabb::AABB_2D) = height(aabb) * width(aabb)
@inline volume(aabb::AABB_3D) = height(aabb) * width(aabb) * depth(aabb)
@inline Base.in(p::Point_2D, aabb::AABB_2D) = aabb.xmin ≤ p[1] ≤ aabb.xmax && 
                                              aabb.ymin ≤ p[2] ≤ aabb.ymax
@inline Base.in(p::Point_3D, aabb::AABB_3D) = aabb.xmin ≤ p[1] ≤ aabb.xmax && 
                                              aabb.ymin ≤ p[2] ≤ aabb.ymax &&
                                              aabb.zmin ≤ p[3] ≤ aabb.zmax

# DEPRECATED. Leaving because there is potential future use
# # Liang-Barsky line clipping algorithm
# # pₖ = 0	            parallel to the clipping boundaries
# # pₖ = 0 and qₖ < 0	    completely outside the boundary
# # pₖ = 0 and qₖ ≥ 0	    inside the parallel clipping boundary
# # pₖ < 0	            line proceeds from outside to inside
# # pₖ > 0	            line proceeds from inside to outside
# function intersect(l::LineSegment_2D{F}, aabb::AABB_2D{F}) where {F <: AbstractFloat}
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
#             return false, SVector(Point_2D{F}(0, 0), Point_2D{F}(0, 0))
#         else # Inside boundaries
#             return true, SVector(Point_2D(l[1].x, aabb.ymin), Point_2D(l[1].x, aabb.ymax))
#         end
#     end
#     if p₃ == 0 # Horizontal line
#         if q₃ < 0 || q₄ < 0 # Outside boundaries
#             return false, SVector(Point_2D{F}(0, 0), Point_2D{F}(0, 0))
#         else # Inside boundaries
#             return true, SVector(Point_2D(aabb.xmin, l[1].y), Point_2D(aabb.xmax, l[1].y))
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
#     t_start < t_stop || return false, SVector(Point_2D{F}(0, 0), Point_2D{F}(0, 0))
# 
#     return true, SVector(l(t_start), l(t_stop))
# end
# 
# Credit to Tavian Barnes (https://tavianator.com/2011/ray_box.html)
# Assumes the line passes all the way through the AABB if it intersects, which is a 
# valid assumption for this ray tracing application. 
function Base.intersect(l::LineSegment_2D, aabb::AABB_2D)
    𝘁₁ = (aabb.origin - l.𝘅₁) ./ l.𝘂
    𝘁₂ = (aabb.corner - l.𝘅₁) ./ l.𝘂

    tmin = max(min(𝘁₁[1], 𝘁₂[1]), min(𝘁₁[2], 𝘁₂[2]))
    tmax = min(max(𝘁₁[1], 𝘁₂[1]), max(𝘁₁[2], 𝘁₂[2]))

    return (tmax >= tmin, SVector(l(tmin), l(tmax)))
end

# A random AABB within [0, 1]ᴺ ⊂ ℝᴺ
# What does the distribution of AABBs look like? Is this uniform? 
function Base.rand(::Type{AABB{N,T}}) where {N,T}
    coord₁ = rand(T, N)
    coord₂ = rand(T, N)
    return AABB{N,T}(Point{N,T}(min.(coord₁, coord₂)), 
                     Point{N,T}(max.(coord₁, coord₂)))  
end

# NB random AABB within [0, 1]ᴺ ⊂ ℝᴺ
function Base.rand(::Type{AABB{N,T}}, NB::Int64) where {N,T}
    return [ rand(AABB{N,T}) for i ∈ 1:NB ]
end

# Return the AABB which contains both bb₁ and bb₂
function Base.union(bb₁::AABB{N,T}, bb₂::AABB{N,T}) where {N,T}
    return AABB(Point{N,T}(min.(bb₁.origin.coord, bb₂.origin.coord)),
                Point{N,T}(max.(bb₁.corner.coord, bb₂.corner.coord)))
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, aabb::AABB_2D)
        p₂ = Point_2D(aabb.xmax, aabb.ymin)
        p₄ = Point_2D(aabb.xmin, aabb.ymax)
        l₁ = LineSegment_2D(aabb.origin, p₂)
        l₂ = LineSegment_2D(p₂, aabb.corner)
        l₃ = LineSegment_2D(aabb.corner, p₄)
        l₄ = LineSegment_2D(p₄, aabb.origin)
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABB_2D})
        point_sets = [convert_arguments(LS, aabb) for aabb in R]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, aabb::AABB_2D)
        p₂ = Point_2D(aabb.xmax, aabb.ymin)
        p₄ = Point_2D(aabb.xmin, aabb.ymax)
        points = [aabb.origin.coord, p₂.coord, aabb.corner.coord, p₄.coord]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, R::Vector{<:AABB_2D})
        points = reduce(vcat, [[aabb.origin.coord, 
                                Point_2D(aabb.xmax, aabb.ymin).coord,
                                aabb.corner.coord, 
                                Point_2D(aabb.xmin, aabb.ymax).coord] for aabb ∈ R])
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
