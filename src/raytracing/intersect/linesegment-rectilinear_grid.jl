function Base.intersect(l::LineSegment{Point{2, T}},
                        g::RectilinearGrid{X, Y, 0, T}) where {X, Y, T}
    # Intersect the bounding box
    # https://tavianator.com/2011/ray_box.html
    ðâ»Â¹ = 1 ./ (l[2] - l[1])
    Xâ = coordinates(l[1])
    râ = ðâ»Â¹ â (Vec(xmin(g), ymin(g)) - Xâ)
    râ = ðâ»Â¹ â (Vec(xmax(g), ymax(g)) - Xâ)
    rmin = maximum(min.(râ, râ))
    rmax = minimum(max.(râ, râ))
    # Valid line clipping
    # Assumes 0 â¤ rmin, rmax â¤ 1
    if rmin â¤ rmax
        pstart = l(rmin)
        pend = l(rmax)
        xlower, xupper = minmax(pstart[1], pend[1])
        ylower, yupper = minmax(pstart[2], pend[2])
        # Get the start and stop indices for the range of grid lines
        # that need to be tested
        # We keep the branches instead of using getsortedfirst, since
        # branched will be pruned at compile time.
        if X â¥ SORTED_ARRAY_THRESHOLD
            xlower_ind = searchsortedfirst(g.x, xlower)
            xupper_ind = searchsortedfirst(g.x, xupper)
        else
            xlower_ind = findsortedfirst(g.x, xlower)
            xupper_ind = findsortedfirst(g.x, xupper)
        end
        if Y â¥ SORTED_ARRAY_THRESHOLD
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
        intersections = Vector{Point{2, T}}(undef, nmax)
        r = ðâ»Â¹[1] * (g.x[xstart] - Xâ[1])
        s = ðâ»Â¹[2] * (g.y[ystart] - Xâ[2])
        i = 2
        j = 2
        k = 1
        # Add the intersection with smallest valid parametric coordinate
        # until we run out of x or y values to test
        while i â¤ nx && j â¤ ny
            if r < s
                if rmin â¤ r â¤ rmax
                    intersections[k] = l(r)
                    k += 1
                end
                r = ðâ»Â¹[1] * (g.x[ixgen[i]] - Xâ[1])
                i += 1
            else
                if rmin â¤ s â¤ rmax
                    intersections[k] = l(s)
                    k += 1
                end
                s = ðâ»Â¹[2] * (g.y[iygen[j]] - Xâ[2])
                j += 1
            end
        end

        # Set the remaining intersections
        while i â¤ nx
            if rmin â¤ r â¤ rmax
                intersections[k] = l(r)
                k += 1
            end
            r = ðâ»Â¹[1] * (g.x[ixgen[i]] - Xâ[1])
            i += 1
        end
        if rmin â¤ r â¤ rmax
            intersections[k] = l(r)
            k += 1
        end

        while j â¤ ny
            if rmin â¤ s â¤ rmax
                intersections[k] = l(s)
                k += 1
            end
            s = ðâ»Â¹[2] * (g.y[iygen[j]] - Xâ[2])
            j += 1
        end
        if rmin â¤ s â¤ rmax
            intersections[k] = l(s)
            k += 1
        end
        # Delete the unused end of the vector
        Base._deleteend!(intersections, nmax - k + 1)
        return intersections
    else
        return Point{2, T}[]
    end
end
