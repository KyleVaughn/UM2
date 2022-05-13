# This version is branchless and is likely faster on the GPU
#function intersect(l::LineSegment, aab::AABox)
#    𝘂⁻¹= 1 ./ l.𝘂   
#    𝘁₁ = 𝘂⁻¹*(aab.minima - l.𝘅₁)
#    𝘁₂ = 𝘂⁻¹*(aab.maxima - l.𝘅₁)
#    tmin = maximum(min.(𝘁₁, 𝘁₂))
#    tmax = minimum(max.(𝘁₁, 𝘁₂))
#    return (tmax >= tmin, SVector(l(tmin), l(tmax)))
#end
function Base.intersect(l::LineSegment{Point{2,T}}, 
                        g::RectilinearGrid{X,Y,0,T}) where {X,Y,T} 
    error("Need to sort points and make unique. Is it faster to do
          that with sort and unique, or should I do so algorithmically?
          Try using two arrays. one for x, one for y. Sort, then just merge the
          two arrays and delete duplicates")
    𝘂 = l[2] - l[1] 
    𝘂⁻¹ = 1 ./ 𝘂
    X₁ = coordinates(l[1])
    # Intersect the bounding box
    # https://tavianator.com/2011/ray_box.html
    r₁ = 𝘂⁻¹ ⊙ (Vec(xmin(g), ymin(g)) - X₁)
    r₂ = 𝘂⁻¹ ⊙ (Vec(xmax(g), ymax(g)) - X₁)
    rmin = maximum(min.(r₁, r₂))
    rmax = minimum(max.(r₁, r₂))
    # Valid line clipping
    # Assumes 0 ≤ rmin, rmax ≤ 1
    if rmin ≤ rmax
        pstart = l(rmin)
        pend = l(rmax)
        xlower, xupper = minmax(pstart[1], pend[1])
        ylower, yupper = minmax(pstart[2], pend[2])
        # Get the start and stop indices for the range of grid lines
        xlower_ind = searchsortedfirst(g.x, xlower)
        xupper_ind = searchsortedfirst(g.x, xupper)
        ylower_ind = searchsortedfirst(g.y, ylower)
        yupper_ind = searchsortedfirst(g.y, yupper)
        # Only intersect the lines that are within the AABB formed by
        # pstart, pend. If the line is oriented in the negative direction, 
        # we need to decrement instead of increment in index
        intersections = Point{2,T}[]
        xstart, xend = xlower_ind, xupper_ind
        xinc = 1
        if pstart[1] > pend[1]
            xend, xstart = xstart, xend
            xinc = -1
        end
        for ix = xstart:xinc:xend
            r = 𝘂⁻¹[1]*(g.x[ix] - X₁[1])
            if 0 ≤ r ≤ 1
                push!(intersections, l(r))
            end
        end

        ystart, yend = ylower_ind, yupper_ind
        yinc = 1
        if pstart[1] > pend[1]
            yend, ystart = ystart, yend
            yinc = -1
        end
        for iy = ystart:yinc:yend
            r = 𝘂⁻¹[2]*(g.y[iy] - X₁[2])
            if 0 ≤ r ≤ 1
                push!(intersections, l(r))
            end
        end
        return intersections
    else
        return Point{2,T}[]
    end
end

# Test line that clips a corner and no other lines
# Tsst line that is vertical or horizontal
