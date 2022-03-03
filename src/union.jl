





# Return the AABox bounding all boxes in the vector 
function Base.union(bbs::Array{AABox{Dim, T}}) where {Dim, T}
    return Base.union(bbs, 1, length(bbs))
end

function Base.union(bbs::Array{AABox{Dim, T}}, lo::Int64, hi::Int64) where {Dim, T}
    if hi-lo === 1
        return Base.union(bbs[lo], bbs[hi])
    elseif hi-lo === 0
        return bbs[lo]
    else    
        mi = Base.Sort.midpoint(lo, hi)
        bb_lo = Base.union(bbs, lo, mi) 
        bb_hi = Base.union(bbs, mi, hi) 
        return Base.union(bb_lo, bb_hi) 
    end     
end
