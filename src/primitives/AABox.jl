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

# Short methods
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
# Intersect
# ---------------------------------------------------------------------------------------------
# Uses a special case of the method in 
# Kay, T. L., & Kajiya, J. T. (1986). Ray tracing complex scenes
#
# Assumes the line passes all the way through the AABox if it intersects, which is a 
# valid assumption for this ray tracing application. 
#
# This version is branchless and is likely faster on the GPU
#function intersect(l::LineSegment, aab::AABox)
#    𝘂⁻¹= 1 ./ l.𝘂   
#    𝘁₁ = 𝘂⁻¹*(aab.origin - l.𝘅₁)
#    𝘁₂ = 𝘂⁻¹*(aab.corner - l.𝘅₁)
#    tmin = maximum(min.(𝘁₁, 𝘁₂))
#    tmax = minimum(max.(𝘁₁, 𝘁₂))
#    return (tmax >= tmin, SVector(l(tmin), l(tmax)))
#end

# This version has branches and is slightly faster on CPU
# Section 5.3.3 in Ericson, C. (2004). Real-time collision detection
function intersect(l::LineSegment{N,T}, aab::AABox{N,T}) where {N,T}
    tmin = typemin(T)
    tmax = typemax(T)
    p_nan = nan_point(typeof(l.𝘅₁)) 
    for i = 1:N
        if abs(l.𝘂[i]) < 1e-6
            # Ray is parallel to slab. No hit if origin not within slab
            if l.𝘅₁[i] < aab.origin[i] || aab.corner[i] < l.𝘅₁[i]
                return (false, SVector(p_nan, p_nan))
            end
        else
            u⁻¹= 1/l.𝘂[i]
            t₁ = (aab.origin[i] - l.𝘅₁[i])*u⁻¹
            t₂ = (aab.corner[i] - l.𝘅₁[i])*u⁻¹
            if t₁ > t₂
                t₁,t₂ = t₂,t₁
            end
            tmin = max(tmin, t₁)
            tmax = min(tmax, t₂)
            if tmin > tmax
                return (false, SVector(p_nan, p_nan))
            end
        end
    end
    return (true, SVector(l(tmin), l(tmax)))
end

# Random
# ---------------------------------------------------------------------------------------------
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

# Union
# ---------------------------------------------------------------------------------------------
# Return the AABox which contains both bb₁ and bb₂
function Base.union(bb₁::AABox{Dim, T}, bb₂::AABox{Dim, T}) where {Dim, T}
    return AABox(Point{Dim, T}(min.(bb₁.origin.coord, bb₂.origin.coord)),
                 Point{Dim, T}(max.(bb₁.corner.coord, bb₂.corner.coord)))
end

# Return the AABox bounding all boxes in the vector 
function Base.union(bbs::Vector{AABox{Dim, T}}) where {Dim, T}
    return Base.union(bbs, 1, length(bbs))
end

function Base.union(bbs::Vector{AABox{Dim, T}}, lo::Int64, hi::Int64) where {Dim, T}
    if hi-lo === 1
        return Base.union(bbs[lo], bbs[hi])
    elseif hi-lo === 0
        return bbs[lo]
    else
        mi = Base.Sort.midpoint(lo, hi) 
        bb_lo = Base.union(bbs, lo, mi)
        bb_hi = Base.union(bbs, mi, hi)
        return Base.union(bb_lo, bb_hi)
    end
end

# Bounding box
# ---------------------------------------------------------------------------------------------
# Bounding box of a vector of points
function boundingbox(points::Vector{<:Point2D})
    xmin = ymin = typemax(T)
    xmax = ymax = typemin(T)
    for i = 1:length(points)
        x,y = points[i].coord  
        if x < xmin
            xmin = x
        end
        if xmax < x
            xmax = x
        end
        if y < ymin
            ymin = y
        end
        if ymax < y
            ymax = y
        end
    end
    return AABox2D(Point2D(xmin, ymin), 
                   Point2D(xmax, ymax))
end

function boundingbox(points::SVector{L, Point2D{T}}) where {L,T} 
    xmin = ymin = typemax(T)
    xmax = ymax = typemin(T)
    for i = 1:L
        x,y = points[i].coord  
        if x < xmin
            xmin = x
        end
        if xmax < x
            xmax = x
        end
        if y < ymin
            ymin = y
        end
        if ymax < y
            ymax = y
        end
    end
    return AABox2D(Point2D(xmin, ymin), 
                   Point2D(xmax, ymax))
end

# Bounding box of a vector of points
function boundingbox(points::Vector{<:Point3D})
    xmin = ymin = zmin = typemax(T)
    xmax = ymax = zmax = typemin(T)
    for i = 1:length(points)
        x,y,z = points[i].coord  
        if x < xmin
            xmin = x
        end
        if xmax < x
            xmax = x
        end
        if y < ymin
            ymin = y
        end
        if ymax < y
            ymax = y
        end
        if z < zmin
            zmin = z
        end
        if zmax < z
            zmax = z
        end
    end
    return AABox3D(Point3D(xmin, ymin, zmin), 
                   Point3D(xmax, ymax, zmax))
end

function boundingbox(points::SVector{L, Point3D{T}}) where {L,T} 
    xmin = ymin = zmin = typemax(T)
    xmax = ymax = zmax = typemin(T)
    for i = 1:L
        x,y,z = points[i].coord  
        if x < xmin
            xmin = x
        end
        if xmax < x
            xmax = x
        end
        if y < ymin
            ymin = y
        end
        if ymax < y
            ymax = y
        end
        if z < zmin
            zmin = z
        end
        if zmax < z
            zmax = z
        end
    end
    return AABox3D(Point3D(xmin, ymin, zmin), 
                   Point3D(xmax, ymax, zmax))
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

    function convert_arguments(LS::Type{<:LineSegments}, aab::AABox3D{T}) where {T}
        #   7----6
        #  /    /|
        # 4----3 |
        # |    | 5
        # |    |/
        # 1----2
        Δx = (aab.xmax - aab.xmin)
        Δy = (aab.ymax - aab.ymin)
        Δz = (aab.zmax - aab.zmin)
        p₁ = aab.origin
        p₂ = Point3D(p₁ + Point3D{T}(Δx,  0,  0)) 
        p₃ = Point3D(p₂ + Point3D{T}( 0, Δy,  0)) 
        p₄ = Point3D(p₁ + Point3D{T}( 0, Δy,  0))
        p₅ = Point3D(p₂ + Point3D{T}( 0,  0, Δz))
        p₆ = aab.corner
        p₇ = Point3D(p₆ - Point3D{T}(Δx,  0,  0))
        p₈ = Point3D(p₁ + Point3D{T}( 0,  0, Δz))
        #       10
        #     +----+
        #   8/   7/|9
        #   +----+ |
        #  4| 3  | +
        #   |   2|/ 5
        #   +----+
        #     1
        l₁  = LineSegment(p₁, p₂)
        l₂  = LineSegment(p₂, p₃)
        l₃  = LineSegment(p₃, p₄)
        l₄  = LineSegment(p₄, p₁)
        l₅  = LineSegment(p₂, p₅)
        l₆  = LineSegment(p₁, p₈)
        l₇  = LineSegment(p₃, p₆)
        l₈  = LineSegment(p₄, p₇)
        l₉  = LineSegment(p₅, p₆)
        l₁₀ = LineSegment(p₆, p₇)
        l₁₁ = LineSegment(p₇, p₈)
        l₁₂ = LineSegment(p₅, p₈)
        lines = [l₁, l₂, l₃, l₄, l₅, l₆, l₇, l₈, l₉, l₁₀, l₁₁, l₁₂]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABox})
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

    function convert_arguments(M::Type{<:Mesh}, aab::AABox3D{T}) where {T}
        Δx = (aab.xmax - aab.xmin)
        Δy = (aab.ymax - aab.ymin)
        Δz = (aab.zmax - aab.zmin)
        p₁ = aab.origin
        p₂ = Point3D(p₁ + Point3D{T}(Δx,  0,  0)) 
        p₃ = Point3D(p₂ + Point3D{T}( 0, Δy,  0)) 
        p₄ = Point3D(p₁ + Point3D{T}( 0, Δy,  0))
        p₅ = Point3D(p₂ + Point3D{T}( 0,  0, Δz))
        p₆ = aab.corner
        p₇ = Point3D(p₆ - Point3D{T}(Δx,  0,  0))
        p₈ = Point3D(p₁ + Point3D{T}( 0,  0, Δz))

        f₁ = Quadrilateral(p₁, p₂, p₃, p₄)
        f₂ = Quadrilateral(p₅, p₆, p₇, p₈)
        f₃ = Quadrilateral(p₂, p₅, p₆, p₃)
        f₄ = Quadrilateral(p₁, p₈, p₇, p₄)
        f₅ = Quadrilateral(p₄, p₃, p₆, p₇)
        f₆ = Quadrilateral(p₁, p₂, p₅, p₈)
        return convert_arguments(M, [f₁, f₂, f₃, f₄, f₅, f₆])
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

    function convert_arguments(M::Type{<:Mesh}, R::Vector{AABox3D{T}}) where {T}
        faces = Quadrilateral3D{T}[]
        for aab ∈ R
            Δx = (aab.xmax - aab.xmin)
            Δy = (aab.ymax - aab.ymin)
            Δz = (aab.zmax - aab.zmin)
            p₁ = aab.origin
            p₂ = Point3D(p₁ + Point3D{T}(Δx,  0,  0)) 
            p₃ = Point3D(p₂ + Point3D{T}( 0, Δy,  0)) 
            p₄ = Point3D(p₁ + Point3D{T}( 0, Δy,  0))
            p₅ = Point3D(p₂ + Point3D{T}( 0,  0, Δz))
            p₆ = aab.corner
            p₇ = Point3D(p₆ - Point3D{T}(Δx,  0,  0))
            p₈ = Point3D(p₁ + Point3D{T}( 0,  0, Δz))
    
            f₁ = Quadrilateral(p₁, p₂, p₃, p₄)
            f₂ = Quadrilateral(p₅, p₆, p₇, p₈)
            f₃ = Quadrilateral(p₂, p₅, p₆, p₃)
            f₄ = Quadrilateral(p₁, p₈, p₇, p₄)
            f₅ = Quadrilateral(p₄, p₃, p₆, p₇)
            f₆ = Quadrilateral(p₁, p₂, p₅, p₈)
            append!(faces, [f₁, f₂, f₃, f₄, f₅, f₆])
        end
        return convert_arguments(M, faces)
    end
end
