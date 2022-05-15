# Like find first, but returns length(v) + 1 in the event that no index
function findsortedfirst(v::AbstractVector, x)
    for i in eachindex(v)
        x ≤ v[i] && return i
    end
    return length(v) + 1
end

function Base.intersect(l::LineSegment{Point{2,T}}, 
                        g::RectilinearGrid{X,Y,0,T}) where {X,Y,T} 
    # Intersect the bounding box
    # https://tavianator.com/2011/ray_box.html
    𝘂⁻¹ = 1 ./ (l[2] - l[1])
    X₁ = coordinates(l[1])
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
        # that need to be tested
        if X ≥ sorted_array_findfirst_threshold
            xlower_ind = searchsortedfirst(g.x, xlower)
            xupper_ind = searchsortedfirst(g.x, xupper)
        else
            xlower_ind = findsortedfirst(g.x, xlower)
            xupper_ind = findsortedfirst(g.x, xupper)
        end
        if Y ≥ sorted_array_findfirst_threshold
            ylower_ind = searchsortedfirst(g.y, ylower)
            yupper_ind = searchsortedfirst(g.y, yupper)
        else
            ylower_ind = findsortedfirst(g.y, ylower)
            yupper_ind = findsortedfirst(g.y, yupper)
        end

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

        # Allocate for the max number of intersections and delete the end 
        # if not needed
        nx = length(ixgen)
        ny = length(iygen)
        nmax = nx + ny
        intersections = Vector{Point{2,T}}(undef, nmax) 
        r = 𝘂⁻¹[1]*(g.x[xstart] - X₁[1])
        s = 𝘂⁻¹[2]*(g.y[ystart] - X₁[2])
        i = 2
        j = 2
        k = 1
        # Add the intersection with smallest valid parametric coordinate
        # until we run out of x or y values to test
        while i ≤ nx && j ≤ ny
            if r < s
                if rmin ≤ r ≤ rmax
                    intersections[k] = l(r)
                    k += 1
                end
                r = 𝘂⁻¹[1]*(g.x[ixgen[i]] - X₁[1])
                i += 1
            else
                if rmin ≤ s ≤ rmax
                    intersections[k] = l(s)
                    k += 1
                end
                s = 𝘂⁻¹[2]*(g.y[iygen[j]] - X₁[2])
                j += 1
            end
        end

        # Set the remaining intersections
        while i ≤ nx
            if rmin ≤ r ≤ rmax
                intersections[k] = l(r)
                k += 1
            end
            r = 𝘂⁻¹[1]*(g.x[ixgen[i]] - X₁[1])
            i += 1
        end
        if rmin ≤ r ≤ rmax
            intersections[k] = l(r)
            k += 1
        end

        while j ≤ ny
            if rmin ≤ s ≤ rmax
                intersections[k] = l(s)
                k += 1
            end
            s = 𝘂⁻¹[2]*(g.y[iygen[j]] - X₁[2])
            j += 1
        end
        if rmin ≤ s ≤ rmax
            intersections[k] = l(s)
            k += 1
        end
        # Delete the unused end of the vector
        Base._deleteend!(intersections, nmax - k + 1)
        return intersections
    else
        return Point{2,T}[]
    end
end
