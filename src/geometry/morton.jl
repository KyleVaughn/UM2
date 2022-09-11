export encode_morton, find_nearest_point

function encode_morton(p::Point2{T}, s_inv::T) where {T}    
    return encode_morton(p[1], p[2], s_inv)    
end  

# annoying needed for anonymous functions
encode_morton(x::UInt32, s_inv::Float32) = x
encode_morton(x::UInt32, s_inv::Float64) = x

function find_nearest_point(
        p::Point2{T}, 
        s_inv::T, 
        points::Vector{Point2{T}}) where {T}
    # Points sorted in z-order are sorted into an implicit quadtree.
    # The nearest point is in the same quadtree node or one of its neighbors.
    # WE ASSUME Z IS 32 BITS and search through the quadtree
    # 
    # ^  2  3     ^  10  11
    # |           |  
    # y  0  1     y  00  01
    #   x -->       x -->
    # 
    # Append 2 bits to the z-order code each level of the tree until we find only one point.
    # The zeroth quadtree node: 00
    # The first quadtree node:  01
    # The second quadtree node: 10
    # The third quadtree node:  11
    z::UInt32 = encode_morton(p, s_inv)
    z_lo = 0x00000000
    z_hi = 0xffffffff
    z_lo_old = z_lo
    z_hi_old = z_hi
    # z_0 = 0b00000000000000000000000000000000 
    z_1 = 0b01000000000000000000000000000000
    z_2 = 0b10000000000000000000000000000000
    # z_3 = 0b11000000000000000000000000000000
    N = length(points)
    i_lo = 1
    i_hi = N 
    for _ in 1:16 # 4^16 = 2^32
        # Find the quadtree node containing z.
        if z < z_lo + z_2 # bottom half 
            z_hi -= z_2
        else # top half
            z_lo += z_2
        end
        if z < z_lo + z_1 # left half
            z_hi -= z_1
        else # right half
            z_lo += z_1
        end
        # Find the lower and upper bound on the indices of the points in the points vector
        # that are in the quadtree node.
        if z_hi !== z_hi_old
            i_hi = searchsortedlast(view(points, i_lo:i_hi), z_hi, by = p->encode_morton(p, s_inv))
            z_hi_old = z_hi
        end
        if z_lo !== z_lo_old
            i_lo = searchsortedfirst(view(points, i_lo:i_hi), z_lo, by = p->encode_morton(p, s_inv))
            z_lo_old = z_lo
        end
        # If the quadtree node contains only one point, return it.
#        if i_lo === i_hi
#            return points[i_lo]
#        end
        # If the quadtree node contains more than one point, continue searching.
        # z_0 >>= 2 
        z_1 >>= 2
        z_2 >>= 2
        #z_3 >>= 2
    end
    error("Failed to find nearest point.")
end
