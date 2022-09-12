export encode_morton, find_nearest_point

function encode_morton(p::Point2{T}, scale_inv::T) where {T}    
    return encode_morton(p[1], p[2], scale_inv)    
end  

# Needed for sort by anonymous function
function encode_morton(uint::U, ::T) where {U <: Unsigned, T <: AbstractFloat}
    return uint
end


function find_nearest_point(
        p::Point2{T}, 
        scale_inv::T, 
        points::Vector{Point2{T}}) where {T}

    # Let x's be points in the points vector.
    # We can find the nearest point by finding the morton curve boundary that
    # contains the point of interest and one of the points in the points vector.
    # We then need to test all points in each of the 8 regions surrounding the
    # boundary to find the nearest point.
    #
    #
    # | -------------- | -------------- | -------------- |
    # |                |                |                |
    # |                |                |                |
    # |         x      |                |                |
    # |                |                |                |
    # |                |                |                |
    # |                |             x  |                |
    # | -------------- | -------------- | -------------- |
    # |                |                |                |
    # |                |           p    |                |
    # |                |                |                |
    # |                |                |                |
    # |                |  x             |                |
    # |                |                |                |
    # | -------------- | -------------- | -------------- |
    # |                |                |                |
    # |                |                |  x             |
    # |                |                |                |
    # |                |                |                |
    # |                |                |                |
    # |                |                |                |
    # | -------------- | -------------- | -------------- |
    #
    #






    # Points sorted in z-order are sorted into an implicit quadtree.
    # The nearest point is in the same quadtree node or one of its neighbors.
    # 
    # ^  2  3     ^  10  11
    # |           |  
    # y  0  1     y  00  01
    #   x -->       x -->
    # 
    # Append 2 bits to the z-order code each level of the tree until 
    # we run out of bits or we run out of points.
    # The zeroth quadtree node: 00
    # The first quadtree node:  01
    # The second quadtree node: 10
    # The third quadtree node:  11
    s = MORTON_MAX_SIDE 
    z = encode_morton(p, scale_inv)
    lo = 1
    N = length(points)
    hi = N
    fixed_lo = false
    fixed_hi = false
    z_search = typemin(MORTON_INDEX_TYPE)
    z_cut = (typemax(MORTON_INDEX_TYPE) >> 1) + 1 # 0x800...000
    all_ones = typemax(MORTON_INDEX_TYPE)
    uzero = zero(MORTON_INDEX_TYPE)
    for l in 1:MORTON_MAX_LEVELS
        fy = z & z_cut; z_cut >>= 1
        fx = z & z_cut; z_cut >>= 1
        z_search |= (fx | fy)
        if (fx !== uzero || fy !== uzero) && !fixed_lo
            # update lo
            view_lo = searchsortedlast(view(points, lo:hi), z_search, 
                                        by=p->encode_morton(p, scale_inv))
            new_lo = lo + view_lo - 1 
            if lo < new_lo && new_lo < hi
                lo = new_lo
            else
                fixed_lo = true
            end
        end
        if (fx === uzero || fy === uzero) && !fixed_hi
            # update hi
            # max morton index is obtained by flipping all lower bits to 1
            z_max = z_search | (all_ones >> (l * 2))
            view_hi = searchsortedfirst(view(points, lo:hi), z_max, 
                                       by=p->encode_morton(p, scale_inv))
            new_hi = lo + view_hi - 1 
            if lo < new_hi && new_hi < hi
                hi = new_hi
            else
                fixed_hi = true
            end
        end

        z_max = z_search | (all_ones >> (l * 2))
        println(lo, " ", hi, " ", z_search, " ",z, " ", z_max)
        println(encode_morton(points[lo], scale_inv), " ", encode_morton(points[hi], scale_inv))
        println(fixed_lo, " ", fixed_hi)
        if fixed_lo && fixed_hi
            break
        end
        linesegments!(LineSegment(points[lo], points[hi]))
        readline()
        pmin = Point2d(decode_morton(z_search, 1/scale_inv))
        pmax = Point2d(decode_morton(z_max, 1/scale_inv))
        linesegments!(AABB(pmin, pmax))
        readline()
    end
    # Find the nearest point in the range [lo, hi]
    nearest = lo
    nearest_dist = distance2(p, points[lo])
    for i in lo+1:hi
        dist = distance2(p, points[i])
        if dist < nearest_dist
            nearest = i
            nearest_dist = dist
        end
    end
    return nearest
end





#        #z_3 >>= 2
#    end
#    error("Failed to find nearest point.")
#end
#
#function find_nearest_point(
#        p::Point2{T}, 
#        scale_inv::T, 
#        points::Vector{Point2{T}}) where {T}
#    # Points sorted in z-order are sorted into an implicit quadtree.
#    # The nearest point is in the same quadtree node or one of its neighbors.
#    # 
#    # ^  2  3     ^  10  11
#    # |           |  
#    # y  0  1     y  00  01
#    #   x -->       x -->
#    # 
#    # Append 2 bits to the z-order code each level of the tree until 
#    # we run out of bits or we run out of points.
#    # The zeroth quadtree node: 00
#    # The first quadtree node:  01
#    # The second quadtree node: 10
#    # The third quadtree node:  11
#    s = MORTON_MAX_SIDE 
#    z = encode_morton(p, scale_inv)
#    lo = 1
#    N = length(points)
#    hi = N
#    fixed_lo = false
#    fixed_hi = false
#    z_search = typemin(MORTON_INDEX_TYPE)
#    z_cut = (typemax(MORTON_INDEX_TYPE) >> 1) + 1 # 0x800...000
#    all_ones = typemax(MORTON_INDEX_TYPE)
#    uzero = zero(MORTON_INDEX_TYPE)
#    for l in 1:MORTON_MAX_LEVELS
#        fy = z & z_cut; z_cut >>= 1
#        fx = z & z_cut; z_cut >>= 1
#        z_search |= (fx | fy)
#        if (fx !== uzero || fy !== uzero) && !fixed_lo
#            # update lo
#            view_lo = searchsortedlast(view(points, lo:hi), z_search, 
#                                        by=p->encode_morton(p, scale_inv))
#            new_lo = lo + view_lo - 1 
#            if lo < new_lo && new_lo < hi
#                lo = new_lo
#            else
#                fixed_lo = true
#            end
#        end
#        if (fx === uzero || fy === uzero) && !fixed_hi
#            # update hi
#            # max morton index is obtained by flipping all lower bits to 1
#            z_max = z_search | (all_ones >> (l * 2))
#            view_hi = searchsortedfirst(view(points, lo:hi), z_max, 
#                                       by=p->encode_morton(p, scale_inv))
#            new_hi = lo + view_hi - 1 
#            if lo < new_hi && new_hi < hi
#                hi = new_hi
#            else
#                fixed_hi = true
#            end
#        end
#
#        z_max = z_search | (all_ones >> (l * 2))
#        println(lo, " ", hi, " ", z_search, " ",z, " ", z_max)
#        println(encode_morton(points[lo], scale_inv), " ", encode_morton(points[hi], scale_inv))
#        println(fixed_lo, " ", fixed_hi)
#        if fixed_lo && fixed_hi
#            break
#        end
#        linesegments!(LineSegment(points[lo], points[hi]))
#        readline()
#        pmin = Point2d(decode_morton(z_search, 1/scale_inv))
#        pmax = Point2d(decode_morton(z_max, 1/scale_inv))
#        linesegments!(AABB(pmin, pmax))
#        readline()
#    end
#    # Find the nearest point in the range [lo, hi]
#    nearest = lo
#    nearest_dist = distance2(p, points[lo])
#    for i in lo+1:hi
#        dist = distance2(p, points[i])
#        if dist < nearest_dist
#            nearest = i
#            nearest_dist = dist
#        end
#    end
#    return nearest
#end
#
#
#
#
#
##        #z_3 >>= 2
##    end
##    error("Failed to find nearest point.")
##end
