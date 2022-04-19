export AABox
export xmin, ymin, zmin, xmax, ymax, zmax, Δx, Δy, Δz  

"""
    AABox(minima::Point{Dim,T}, maxima::Point{Dim,T})

A `Dim`-dimensional axis-aligned bounding box using two `Dim`-dimensional
points, representing the `minima` and `maxima` of the box. 
"""
struct AABox{Dim,T}
    minima::Point{Dim,T}
    maxima::Point{Dim,T}
    function AABox{Dim,T}(minima::Point{Dim,T}, maxima::Point{Dim,T}) where {Dim,T}
        for i ∈ 1:Dim
            if maxima[i] ≤ minima[i]
                error("AABox maxima must be greater that the minima")
            end
        end
        return new{Dim,T}(minima, maxima)
    end
end

# constructors
function AABox(minima::Point{Dim,T}, maxima::Point{Dim,T}) where {Dim,T}
    return AABox{Dim,T}(minima, maxima)
end
AABox(minima, maxima) = AABox(Point(minima), Point(maxima))

Base.isapprox(aab₁::AABox, aab₂::AABox) = aab₁.minima ≈ aab₂.minima && 
                                          aab₁.maxima ≈ aab₂.maxima 
xmin(aab::AABox) = aab.minima[1] 
ymin(aab::AABox) = aab.minima[2] 
zmin(aab::AABox) = aab.minima[3] 
xmax(aab::AABox) = aab.maxima[1] 
ymax(aab::AABox) = aab.maxima[2] 
zmax(aab::AABox) = aab.maxima[3] 
Δx(aab::AABox) = xmax(aab) - xmin(aab) 
Δy(aab::AABox) = ymax(aab) - ymin(aab)
Δz(aab::AABox) = zmax(aab) - zmin(aab)

function Base.show(io::IO, aab::AABox)
    print(io, "AABox($(aab.minima), $(aab.maxima))")
end
