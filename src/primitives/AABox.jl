"""
    AABox(minima::Point{Dim, T}, maxima::Point{Dim, T})

Construct a `Dim`-dimensional axis-aligned bounding box using two Dim`-dimensional
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

@inline Δx(aab::AABox) = aab.xmax - aab.xmin
@inline Δy(aab::AABox) = aab.ymax - aab.ymin
@inline Δz(aab::AABox) = aab.zmax - aab.zmin
