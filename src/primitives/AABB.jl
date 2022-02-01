# Axis-aligned bounding box.
# A Dim-dimensional box requires 2 Dim-dimensional points to specify the boundary:
#   One point to specify the box origin, and one to specify the opposite (furthest corner)
struct AABB{Dim, T}
    origin::Point{Dim, T}
    corner::Point{Dim, T}
end

const AABB2D = AABB{2}

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
AABB{Dim}(p₁::Point{Dim, T}, p₂::Point{Dim, T}) where {Dim, T} = AABB{Dim, T}(p₁, p₂)

# Methods
# ---------------------------------------------------------------------------------------------
@inline width(aabb::AABB) = aabb.xmax - aabb.xmin
@inline height(aabb::AABB) = aabb.ymax - aabb.ymin
@inline depth(aabb::AABB) = aabb.ymax - aabb.ymin
@inline area(aabb::AABB2D) = height(aabb) * width(aabb)
@inline Base.in(p::Point2D, aabb::AABB2D) = aabb.xmin ≤ p[1] ≤ aabb.xmax && 
                                            aabb.ymin ≤ p[2] ≤ aabb.ymax

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

# A random AABB within the Dim-dimensional unit hypercube 
# What does the distribution of AABBs look like? Is this uniform? 
function Base.rand(::Type{AABB{Dim, T}}) where {Dim, T}
    coord₁ = rand(T, Dim)
    coord₂ = rand(T, Dim)
    return AABB{Dim, T}(Point{Dim, T}(min.(coord₁, coord₂)), 
                        Point{Dim, T}(max.(coord₁, coord₂)))  
end

# N random AABBs within the Dim-dimensional unit hypercube 
function Base.rand(::Type{AABB{Dim, T}}, num_boxes::Int64) where {Dim, T}
    return [ rand(AABB{Dim, T}) for i ∈ 1:num_boxes ]
end

# Return the AABB which contains both bb₁ and bb₂
function Base.union(bb₁::AABB{Dim, T}, bb₂::AABB{Dim, T}) where {Dim, T}
    return AABB(Point{Dim, T}(min.(bb₁.origin.coord, bb₂.origin.coord)),
                Point{Dim, T}(max.(bb₁.corner.coord, bb₂.corner.coord)))
end

# Bounding box
# ---------------------------------------------------------------------------------------------
# Bounding box of a vector of points
function boundingbox(points::Vector{<:Point2D})
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return AABB2D(Point2D(minimum(x), minimum(y)), Point2D(maximum(x), maximum(y)))
end

# Bounding box of a vector of points
function boundingbox(points::SVector{L, Point2D}) where {L} 
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return AABB2D(Point2D(minimum(x), minimum(y)), Point2D(maximum(x), maximum(y)))
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
