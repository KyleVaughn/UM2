"""
    AABox(minima::Point{Dim, T}, maxima::Point{Dim, T})

Construct a `Dim`-dimensional axis-aligned bounding box using two `Dim`-dimensional
points, representing the minima and maxima of the box. 
"""
struct AABox{Dim, T}
    minima::Point{Dim, T}
    maxima::Point{Dim, T}
end

const AABox2D = AABox{2}
const AABox3D = AABox{3}

function Base.getproperty(aab::AABox, sym::Symbol)
    if sym === :xmin
        return aab.minima[1]
    elseif sym === :ymin
        return aab.minima[2]
    elseif sym === :zmin
        return aab.minima[3]
    elseif sym === :xmax
        return aab.maxima[1]
    elseif sym === :ymax
        return aab.maxima[2]
    elseif sym === :zmax
        return aab.maxima[3]
    else # fallback to getfield
        return getfield(aab, sym)
    end
end

AABox{Dim}(p₁::Point{Dim, T}, p₂::Point{Dim, T}) where {Dim, T} = AABox{Dim, T}(p₁, p₂)
# AABox3D from 2 NTuple{3} points
function AABox{3, T}(p₁::NTuple{3}, p₂::NTuple{3}) where {T}
    return AABox{3, T}(Point3D{T}(p₁[1], p₁[2], p₁[3]), 
                       Point3D{T}(p₂[1], p₂[2], p₂[3]))
end
function AABox{3}(p₁::NTuple{3, T}, p₂::NTuple{3, T}) where {T}
    return AABox{3, T}(Point3D{T}(p₁[1], p₁[2], p₁[3]), 
                       Point3D{T}(p₂[1], p₂[2], p₂[3]))
end
function AABox(p₁::NTuple{3, T}, p₂::NTuple{3, T}) where {T}
    return AABox{3, T}(Point3D{T}(p₁[1], p₁[2], p₁[3]), 
                       Point3D{T}(p₂[1], p₂[2], p₂[3]))
end
# AABox2D from 2 NTuple{2} points
function AABox{2, T}(p₁::NTuple{2}, p₂::NTuple{2}) where {T}
    return AABox{2, T}(Point(p₁[1], p₁[2]), 
                       Point(p₂[1], p₂[2]))
end
function AABox{2}(p₁::NTuple{2, T}, p₂::NTuple{2, T}) where {T}
    return AABox{2, T}(Point(p₁[1], p₁[2]), 
                       Point(p₂[1], p₂[2]))
end
function AABox(p₁::NTuple{2, T}, p₂::NTuple{2, T}) where {T}
    return AABox{2, T}(Point(p₁[1], p₁[2]), 
                       Point(p₂[1], p₂[2]))
end
# AABox from minima and maxima 
function AABox{2, T}(xmin, ymin, xmax, ymax) where {T}
    if xmax ≤ xmin || ymax ≤ ymin
        error("Invalid AABox extrema")
    end
    return AABox{2, T}(Point2D{T}(xmin, ymin), Point2D{T}(xmax, ymax))
end
function AABox{2}(xmin::T, ymin::T, xmax::T, ymax::T) where {T}
    if xmax ≤ xmin || ymax ≤ ymin
        error("Invalid AABox extrema")
    end
    return AABox{2, T}(Point2D{T}(xmin, ymin), Point2D{T}(xmax, ymax))
end
function AABox(xmin::T, ymin::T, xmax::T, ymax::T) where {T}
    if xmax ≤ xmin || ymax ≤ ymin
        error("Invalid AABox extrema")
    end
    return AABox{2, T}(Point2D{T}(xmin, ymin), Point2D{T}(xmax, ymax))
end

# AABox with minima at (0, 0)
function AABox{2, T}(xmax::N1, ymax::N1) where {T, N1 <: Number, N2 <: Number}
    if xmax ≤ 0 || ymax ≤ 0
        error("Invalid AABox extrema")
    end
    return AABox{2, T}(Point2D{T}(0, 0), Point2D{T}(xmax, ymax))
end
function AABox{2}(xmax::T, ymax::T) where {T<:Number}
    if xmax ≤ 0 || ymax ≤ 0
        error("Invalid AABox extrema")
    end
    return AABox{2, T}(Point2D{T}(0, 0), Point2D{T}(xmax, ymax))
end
function AABox(xmax::T, ymax::T) where {T<:Number}
    if xmax ≤ 0 || ymax ≤ 0
        error("Invalid AABox extrema")
    end
    return AABox{2, T}(Point2D{T}(0, 0), Point2D{T}(xmax, ymax))
end

@inline Δx(aab::AABox) = aab.xmax - aab.xmin
@inline Δy(aab::AABox) = aab.ymax - aab.ymin
@inline Δz(aab::AABox) = aab.zmax - aab.zmin

function split(aab::AABox2D{T}, xdiv::SVector{X, T}, ydiv::SVector{Y, T}
              ) where {T, X, Y}
    if any(y->y < aab.ymin || aab.ymax < y, ydiv)
        error("y-coordinate divisions must be inside the AABox")
    end
    if any(x->x < aab.xmin || aab.xmax < x, xdiv)
        error("x-coordinate divisions must be inside the AABox")
    end
    
    if Y === 0
        y_boxes = MVector(aab)
    else
        y_boxes = MVector{Y+1, AABox2D{T}}(undef) 
        ydiv_sorted = sort(ydiv)
        ymin = aab.ymin
        xmin = aab.xmin
        xmax = aab.xmax
        for i ∈ 1:Y
            ymax = ydiv_sorted[i]
            y_boxes[i] = AABox2D(Point2D(xmin, ymin), Point2D(xmax, ymax))
            ymin = ymax
        end
        y_boxes[Y+1] = AABox2D(Point2D(xmin, ymin), Point2D(xmax, aab.ymax))
    end

    if X === 0
        return SMatrix{1, Y+1, AABox2D{T}, Y+1}(y_boxes.data)
    else
        Nbox = (X+1)*(Y+1)
        ibox = 0
        boxes = MMatrix{X+1, Y+1, AABox2D{T}, Nbox}(undef) 
        xdiv_sorted = sort(xdiv)
        for j ∈ 1:Y+1
            ymin = y_boxes[j].ymin  
            ymax = y_boxes[j].ymax
            xmin = aab.xmin
            for i ∈ 1:X
                xmax = xdiv_sorted[i]
                boxes[i, j] = AABox2D(Point2D(xmin, ymin), Point2D(xmax, ymax))
                xmin = xmax
            end
                boxes[X+1, j] = AABox2D(Point2D(xmin, ymin), Point2D(aab.xmax, ymax))
        end
        return SMatrix(boxes)
    end
end

function split(aab::AABox2D{BigFloat}, xdiv::SVector{X, BigFloat}, 
                                       ydiv::SVector{Y, BigFloat}) where {X, Y}
    if any(y->y < aab.ymin || aab.ymax < y, ydiv)
        error("y-coordinate divisions must be inside the AABox")
    end
    if any(x->x < aab.xmin || aab.xmax < x, xdiv)
        error("x-coordinate divisions must be inside the AABox")
    end
    
    if Y === 0
        y_boxes = [aab]
    else
        y_boxes = Vector{AABox2D{BigFloat}}(undef, Y+1) 
        ydiv_sorted = sort(ydiv)
        ymin = aab.ymin
        xmin = aab.xmin
        xmax = aab.xmax
        for i ∈ 1:Y
            ymax = ydiv_sorted[i]
            y_boxes[i] = AABox2D(Point2D(xmin, ymin), Point2D(xmax, ymax))
            ymin = ymax
        end
        y_boxes[Y+1] = AABox2D(Point2D(xmin, ymin), Point2D(xmax, aab.ymax))
    end

    if X === 0
        boxes = Matrix{AABox2D{BigFloat}}(undef, 1, Y+1)
        for i ∈ 1:Y+1
            boxes[i] = y_boxes[i]
        end
        return boxes
    else
        ibox = 0
        boxes = Matrix{AABox2D{BigFloat}}(undef, X+1, Y+1) 
        xdiv_sorted = sort(xdiv)
        for j ∈ 1:Y+1
            ymin = y_boxes[j].ymin  
            ymax = y_boxes[j].ymax
            xmin = aab.xmin
            for i ∈ 1:X
                xmax = xdiv_sorted[i]
                boxes[i, j] = AABox2D(Point2D(xmin, ymin), Point2D(xmax, ymax))
                xmin = xmax
            end
                boxes[X+1, j] = AABox2D(Point2D(xmin, ymin), Point2D(aab.xmax, ymax))
        end
        return boxes
    end
end

function split(aab::AABox2D{T}, xdiv::Vector{T}, ydiv::Vector{T}) where {T}
    return split(aab, SVector{length(xdiv), T}(xdiv), SVector{length(ydiv), T}(ydiv))
end
