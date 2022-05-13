function Base.intersect(l::LineSegment{Point{2,T}}, 
                        g::RectilinearGrid{X,Y,0,T}) where {X,Y,T} 
    𝘂⁻¹ = 1 ./ (l[2] - l[1])
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
        xstart, xend = xlower_ind, xupper_ind
        xinc = 1
        if pstart[1] > pend[1]
            xend, xstart = xstart, xend
            xinc = -1
        end
        ixgen = xstart:xinc:xend

        ystart, yend = ylower_ind, yupper_ind
        yinc = 1
        if pstart[2] > pend[2]
            yend, ystart = ystart, yend
            yinc = -1
        end
        iygen = ystart:yinc:yend

        intersections = Point{2,T}[] 
        nx = length(ixgen)
        ny = length(iygen)
        r = 𝘂⁻¹[1]*(g.x[xstart] - X₁[1])
        s = 𝘂⁻¹[2]*(g.y[ystart] - X₁[2])
        i = 2
        j = 2
        while i ≤ nx && j ≤ ny
            if r < s
                if rmin ≤ r ≤ rmax
                    push!(intersections, l(r))
                end
                r = 𝘂⁻¹[1]*(g.x[ixgen[i]] - X₁[1])
                i += 1
            else
                if rmin ≤ s ≤ rmax
                    push!(intersections, l(s))
                end
                s = 𝘂⁻¹[2]*(g.y[iygen[j]] - X₁[2])
                j += 1
            end
        end

        while i ≤ nx
            if rmin ≤ r ≤ rmax
                push!(intersections, l(r))
            end
            r = 𝘂⁻¹[1]*(g.x[ixgen[i]] - X₁[1])
            i += 1
        end
        if rmin ≤ r ≤ rmax
            push!(intersections, l(r))
        end

        while j ≤ ny
            if rmin ≤ s ≤ rmax
                push!(intersections, l(s))
            end
            s = 𝘂⁻¹[2]*(g.y[iygen[j]] - X₁[2])
            j += 1
        end
        if rmin ≤ s ≤ rmax
            push!(intersections, l(s))
        end

        return intersections
    else
        return Point{2,T}[]
    end
end

# Test line that clips a corner and no other lines
# Tsst line that is vertical or horizontal
# 45 degree angle
