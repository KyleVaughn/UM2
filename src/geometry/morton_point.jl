export morton_encode,
       morton_sort!,
       morton_sortperm,
       morton_neighbors

function morton_encode(p::Point2{T}, scale_inv::T) where {T}    
    return normalized_morton_encode(p[1], p[2], scale_inv)    
end  

function morton_sort!(points::Vector{Point2{T}}, scale_inv::T) where {T}
    return sort!(points, by=p->morton_encode(p, scale_inv))
end

function morton_sortperm(points::Vector{Point2{T}}, scale_inv::T) where {T}
    return sortperm(points, by=p->morton_encode(p, scale_inv))
end

# Return the index of the point whose z-order bounds the given point from below and above
function morton_neighbors(p::Point2{T}, points::Vector{Point2{T}}, scale_inv::T) where {T}
    # Binary search on the z-order of the points
    hi = searchsortedfirst(points, p, by=p->morton_encode(p, scale_inv))
    npts = length(points)
    if hi == 1
        return (1, 1)
    elseif hi == npts + 1
        return (npts, npts)
    else
        return (hi-1, hi)
    end
end
