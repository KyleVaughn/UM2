# Axis-aligned box.
# A Dim-dimensional box requires 2 Dim-dimensional points to specify the boundary:
#   One point to specify the box origin, and one to specify the opposite (furthest corner)
struct AABox{Dim, T}
    origin::Point{Dim, T}
    corner::Point{Dim, T}
end

const AABox2D = AABox{2}
const AABox3D = AABox{3}

function Base.getproperty(aab::AABox, sym::Symbol)
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
AABox{Dim}(p₁::Point{Dim, T}, p₂::Point{Dim, T}) where {Dim, T} = AABox{Dim, T}(p₁, p₂)

# Methods
# ---------------------------------------------------------------------------------------------
@inline width(aab::AABox)  = aab.xmax - aab.xmin
@inline height(aab::AABox) = aab.ymax - aab.ymin
@inline depth(aab::AABox)  = aab.ymax - aab.ymin
@inline area(aab::AABox2D) = height(aab) * width(aab)
@inline function area(aab::AABox3D)
            x = width(aab); y = height(aab); z = depth(aab);
            return 2(x*z + y*z + x*y)
        end
@inline volume(aab::AABox3D) = height(aab) * width(aab) * depth(aab)
@inline Base.in(p::Point2D, aab::AABox2D) = aab.xmin ≤ p[1] ≤ aab.xmax && 
                                            aab.ymin ≤ p[2] ≤ aab.ymax
@inline Base.in(p::Point3D, aab::AABox3D) = aab.xmin ≤ p[1] ≤ aab.xmax && 
                                            aab.ymin ≤ p[2] ≤ aab.ymax &&
                                            aab.zmin ≤ p[3] ≤ aab.zmax

# Using the slab method
# Assumes the line passes all the way through the AABox if it intersects, which is a 
# valid assumption for this ray tracing application. 
function Base.intersect(l::LineSegment, aab::AABox)
    𝘂⁻¹= 1 ./ l.𝘂   
    𝘁₁ = 𝘂⁻¹*(aab.origin - l.𝘅₁)
    𝘁₂ = 𝘂⁻¹*(aab.corner - l.𝘅₁)
    tmin = maximum(min.(𝘁₁, 𝘁₂))
    tmax = minimum(max.(𝘁₁, 𝘁₂))
    return (tmax >= tmin, SVector(l(tmin), l(tmax)))
end

# A random AABox within the Dim-dimensional unit hypercube 
# What does the distribution of AABoxs look like? Is this uniform? 
function Base.rand(::Type{AABox{Dim, T}}) where {Dim, T}
    coord₁ = rand(T, Dim)
    coord₂ = rand(T, Dim)
    return AABox{Dim, T}(Point{Dim, T}(min.(coord₁, coord₂)), 
                         Point{Dim, T}(max.(coord₁, coord₂)))  
end

# N random AABoxs within the Dim-dimensional unit hypercube 
function Base.rand(::Type{AABox{Dim, T}}, num_boxes::Int64) where {Dim, T}
    return [ rand(AABox{Dim, T}) for i ∈ 1:num_boxes ]
end

# Return the AABox which contains both bb₁ and bb₂
function Base.union(bb₁::AABox{Dim, T}, bb₂::AABox{Dim, T}) where {Dim, T}
    return AABox(Point{Dim, T}(min.(bb₁.origin.coord, bb₂.origin.coord)),
                 Point{Dim, T}(max.(bb₁.corner.coord, bb₂.corner.coord)))
end

# Bounding box
# ---------------------------------------------------------------------------------------------
# Bounding box of a vector of points
function boundingbox(points::Vector{<:Point2D})
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return AABox2D(Point2D(minimum(x), minimum(y)), Point2D(maximum(x), maximum(y)))
end

# Bounding box of a vector of points
function boundingbox(points::SVector{L, Point2D}) where {L} 
    x = getindex.(points, 1)
    y = getindex.(points, 2)
    return AABox2D(Point2D(minimum(x), minimum(y)), Point2D(maximum(x), maximum(y)))
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, aab::AABox2D)
        p₂ = Point2D(aab.xmax, aab.ymin)
        p₄ = Point2D(aab.xmin, aab.ymax)
        l₁ = LineSegment2D(aab.origin, p₂)
        l₂ = LineSegment2D(p₂, aab.corner)
        l₃ = LineSegment2D(aab.corner, p₄)
        l₄ = LineSegment2D(p₄, aab.origin)
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABox2D})
        point_sets = [convert_arguments(LS, aab) for aab in R]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, aab::AABox2D)
        p₂ = Point2D(aab.xmax, aab.ymin)
        p₄ = Point2D(aab.xmin, aab.ymax)
        points = [aab.origin.coord, p₂.coord, aab.corner.coord, p₄.coord]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, R::Vector{<:AABox2D})
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
