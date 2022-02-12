# Axis-aligned box.
# A Dim-dimensional box requires 2 Dim-dimensional points to specify the boundary:
#   One point to specify the box origin, and one to specify the opposite (furthest corner)
struct AAB{Dim, T}
    origin::Point{Dim, T}
    corner::Point{Dim, T}
end

const AAB2D = AAB{2}
const AAB3D = AAB{3}

function Base.getproperty(aab::AAB, sym::Symbol)
    if sym === :xmin
        return aab.origin[1]
    elseif sym === :ymin
        return aab.origin[2]
    elseif sym === :zmin
        return aab.origin[3]
    elseif sym === :xmax
        return aab.corner[1]
    elseif sym === :ymax
        return aab.corner[2]
    elseif sym === :zmax
        return aab.corner[3]
    else # fallback to getfield
        return getfield(aab, sym)
    end
end

# Constructors
# ---------------------------------------------------------------------------------------------
AAB{Dim}(p₁::Point{Dim, T}, p₂::Point{Dim, T}) where {Dim, T} = AAB{Dim, T}(p₁, p₂)

# Methods
# ---------------------------------------------------------------------------------------------
@inline width(aab::AAB)  = aab.xmax - aab.xmin
@inline height(aab::AAB) = aab.ymax - aab.ymin
@inline depth(aab::AAB)  = aab.ymax - aab.ymin
@inline area(aab::AAB2D) = height(aab) * width(aab)
@inline Base.in(p::Point2D, aab::AAB2D) = aab.xmin ≤ p[1] ≤ aab.xmax && 
                                          aab.ymin ≤ p[2] ≤ aab.ymax

# Credit to Tavian Barnes (https://tavianator.com/2011/ray_box.html)
# Assumes the line passes all the way through the AAB if it intersects, which is a 
# valid assumption for this ray tracing application. 
function Base.intersect(l::LineSegment2D, aab::AAB2D)
    𝘁₁ = (aab.origin - l.𝘅₁) ./ l.𝘂
    𝘁₂ = (aab.corner - l.𝘅₁) ./ l.𝘂

    tmin = max(min(𝘁₁[1], 𝘁₂[1]), min(𝘁₁[2], 𝘁₂[2]))
    tmax = min(max(𝘁₁[1], 𝘁₂[1]), max(𝘁₁[2], 𝘁₂[2]))

    return (tmax >= tmin, SVector(l(tmin), l(tmax)))
end

# A random AAB within the Dim-dimensional unit hypercube 
# What does the distribution of AABs look like? Is this uniform? 
function Base.rand(::Type{AAB{Dim, T}}) where {Dim, T}
    coord₁ = rand(T, Dim)
    coord₂ = rand(T, Dim)
    return AAB{Dim, T}(Point{Dim, T}(min.(coord₁, coord₂)), 
                        Point{Dim, T}(max.(coord₁, coord₂)))  
end

# N random AABs within the Dim-dimensional unit hypercube 
function Base.rand(::Type{AAB{Dim, T}}, num_boxes::Int64) where {Dim, T}
    return [ rand(AAB{Dim, T}) for i ∈ 1:num_boxes ]
end

# Return the AAB which contains both bb₁ and bb₂
function Base.union(bb₁::AAB{Dim, T}, bb₂::AAB{Dim, T}) where {Dim, T}
    return AAB(Point{Dim, T}(min.(bb₁.origin.coord, bb₂.origin.coord)),
                Point{Dim, T}(max.(bb₁.corner.coord, bb₂.corner.coord)))
end

# Bounding box
# ---------------------------------------------------------------------------------------------
# Bounding box of a vector of points
function boundingbox(points::Vector{<:Point2D})
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return AAB2D(Point2D(minimum(x), minimum(y)), Point2D(maximum(x), maximum(y)))
end

# Bounding box of a vector of points
function boundingbox(points::SVector{L, Point2D}) where {L} 
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return AAB2D(Point2D(minimum(x), minimum(y)), Point2D(maximum(x), maximum(y)))
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, aab::AAB2D)
        p₂ = Point2D(aab.xmax, aab.ymin)
        p₄ = Point2D(aab.xmin, aab.ymax)
        l₁ = LineSegment2D(aab.origin, p₂)
        l₂ = LineSegment2D(p₂, aab.corner)
        l₃ = LineSegment2D(aab.corner, p₄)
        l₄ = LineSegment2D(p₄, aab.origin)
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AAB2D})
        point_sets = [convert_arguments(LS, aab) for aab in R]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, aab::AAB2D)
        p₂ = Point2D(aab.xmax, aab.ymin)
        p₄ = Point2D(aab.xmin, aab.ymax)
        points = [aab.origin.coord, p₂.coord, aab.corner.coord, p₄.coord]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, R::Vector{<:AAB2D})
        points = reduce(vcat, [[aab.origin.coord, 
                                Point2D(aab.xmax, aab.ymin).coord,
                                aab.corner.coord, 
                                Point2D(aab.xmin, aab.ymax).coord] for aab ∈ R])
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
