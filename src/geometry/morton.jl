export encode_morton,
       sort_morton_order!,
       sortperm_morton_order, 
       morton_z_neighbors

function encode_morton(p::Point2{T}, scale_inv::T) where {T}    
    return normalized_encode_morton(p[1], p[2], scale_inv)    
end  

function sort_morton_order!(points::Vector{Point2{T}}, scale_inv::T) where {T}
    return sort!(points, by=p->encode_morton(p, scale_inv))
end

function sortperm_morton_order(points::Vector{Point2{T}}, scale_inv::T) where {T}
    return sortperm(points, by=p->encode_morton(p, scale_inv))
end

# Return the index of the point whose z-order bounds the given point from below and above
function morton_z_neighbors(p::Point2{T}, points::Vector{Point2{T}}, scale_inv::T) where {T}
    # Binary search on the z-order of the points
    hi = searchsortedfirst(points, p, by=p->encode_morton(p, scale_inv))
    lo = hi - 1
    npts = length(points)
    if npts < hi 
        hi = npts
    end
    if lo === 0
        lo = 1
    end
    return lo, hi
end
