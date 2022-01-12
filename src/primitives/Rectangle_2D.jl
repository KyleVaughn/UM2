# Axis-aligned rectangle in 2D
struct Rectangle_2D{F <: AbstractFloat} <: Face_2D{F}
    bl::Point_2D{F} # Bottom left point
    tr::Point_2D{F} # Top right point
end

# Base
# -------------------------------------------------------------------------------------------------
# Note: all branches but the correct one are pruned by the compiler
function getproperty(rect::Rectangle_2D, sym::Symbol)
    if sym === :xmin
        return rect.bl.x
    elseif sym === :ymin
        return rect.bl.y
    elseif sym === :xmax
        return rect.tr.x
    elseif sym === :ymax
        return rect.tr.y
    else # fallback to getfield
        return getfield(rect, sym)
    end
end

# Methods
# -------------------------------------------------------------------------------------------------
@inline width(rect::Rectangle_2D) = rect.tr.x - rect.bl.x
@inline height(rect::Rectangle_2D) = rect.tr.y - rect.bl.y
@inline area(rect::Rectangle_2D) = height(rect) * width(rect)
@inline in(p::Point_2D, rect::Rectangle_2D) = rect.xmin ≤ p.x ≤ rect.xmax && rect.ymin ≤ p.y ≤ rect.ymax

# Liang-Barsky line clipping algorithm
# pₖ = 0	            parallel to the clipping boundaries
# pₖ = 0 and qₖ < 0	    completely outside the boundary
# pₖ = 0 and qₖ ≥ 0	    inside the parallel clipping boundary
# pₖ < 0	            line proceeds from outside to inside
# pₖ > 0	            line proceeds from inside to outside
function intersect(l::LineSegment_2D{F}, rect::Rectangle_2D{F}) where {F <: AbstractFloat}
    p₂ = l[2].x - l[1].x
    p₁ = -p₂
    p₄ = l[2].y - l[1].y
    p₃ = -p₄

    q₁ = l[1].x - rect.xmin
    q₂ = rect.xmax - l[1].x
    q₃ = l[1].y - rect.ymin
    q₄ = rect.ymax - l[1].y

    # Line parallel to clipping window
    if p₁ == 0 # Vertical line
        if q₁ < 0 || q₂ < 0 # Outside boundaries
            return false, SVector(Point_2D{F}(0, 0), Point_2D{F}(0, 0))
        else # Inside boundaries
            return true, SVector(Point_2D(l[1].x, rect.ymin), Point_2D(l[1].x, rect.ymax))
        end
    end
    if p₃ == 0 # Horizontal line
        if q₃ < 0 || q₄ < 0 # Outside boundaries
            return false, SVector(Point_2D{F}(0, 0), Point_2D{F}(0, 0))
        else # Inside boundaries
            return true, SVector(Point_2D(rect.xmin, l[1].y), Point_2D(rect.xmax, l[1].y))
        end
    end

    t₁ = q₁ / p₁
    t₂ = q₂ / p₂
    if (p₁ < 0)
        t_min2 = t₁
        t_max2 = t₂
    else
        t_min2 = t₂
        t_max2 = t₁
    end

    t₃ = q₃ / p₃
    t₄ = q₄ / p₄
    if (p₃ < 0)
        t_min3 = t₃
        t_max3 = t₄
    else
        t_min3 = t₄
        t_max3 = t₃
    end

    t_start = max(F(0), t_min2, t_min3)
    t_stop = min(F(1), t_max2, t_max3)

    # Line outside clipping window
    t_start < t_stop || return false, SVector(Point_2D{F}(0, 0), Point_2D{F}(0, 0))

    return true, SVector(l(t_start), l(t_stop))
end

function union(r₁::Rectangle_2D, r₂::Rectangle_2D)
    xmin = min(r₁.xmin, r₂.xmin)
    ymin = min(r₁.ymin, r₂.ymin)
    xmax = max(r₁.xmax, r₂.xmax)
    ymax = max(r₁.ymax, r₂.ymax)
    return Rectangle_2D(Point_2D(xmin, ymin), Point_2D(xmax, ymax))
end

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, rect::Rectangle_2D)
        p₂ = Point_2D(rect.xmax, rect.ymin)
        p₄ = Point_2D(rect.xmin, rect.ymax)
        l₁ = LineSegment_2D(rect.bl, p₂)
        l₂ = LineSegment_2D(p₂, rect.tr)
        l₃ = LineSegment_2D(rect.tr, p₄)
        l₄ = LineSegment_2D(p₄, rect.bl)
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:Rectangle_2D})
        point_sets = [convert_arguments(LS, rect) for rect in R]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, rect::Rectangle_2D)
        p₂ = Point_2D(rect.xmax, rect.ymin)
        p₄ = Point_2D(rect.xmin, rect.ymax)
        points = [rect.bl, p₂, rect.tr, p₄]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, R::Vector{<:Rectangle_2D})
        points = reduce(vcat, [[rect.bl, Point_2D(rect.xmax, rect.ymin),
                                rect.tr, Point_2D(rect.xmin, rect.ymax)] for rect ∈ R])
        faces = zeros(Int64, 2*length(Q), 3)
        j = 0
        for i in 1:2:2*length(Q)
            faces[i    , :] = [1 2 3] .+ j
            faces[i + 1, :] = [3 4 1] .+ j
            j += 4
        end
        return convert_arguments(M, points, faces)
    end
end
