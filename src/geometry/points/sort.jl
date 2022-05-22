# Sort points based on their distance from another point.
# Default algorithm is quicksort. If the vector is less than 20 elements, insertion sort
# is used instead.
# This is essentially Julia's Base.Sort

defalg(v::Vector{<:Point}) = Base.Sort.QuickSort
function Base.sort!(p::Point, 
                    v::Vector{<:Point}; 
                    alg::Base.Sort.Algorithm=defalg(v), 
                    order::Base.Ordering=Base.Forward)
    sort!(p, v, firstindex(v), lastindex(v), alg, order)
end

function Base.sort(p::Point, 
                   v::Vector{<:Point};
                   alg::Base.Sort.Algorithm=defalg(v), 
                   order::Base.Ordering=Base.Forward)
    v2 = similar(v)
    @. v2 = v
    sort!(p, v2, firstindex(v2), lastindex(v2), alg, order)
    return v2
end

function Base.sort!(p::Point, 
                    v::Vector{<:Point}, 
                    lo::Integer, 
                    hi::Integer,
                    ::Base.Sort.InsertionSortAlg, 
                    o::Base.Ordering)
    @inbounds for i ∈ lo+1:hi
        j = i
        d = distance²(p, v[i])
        pt = v[i]
        while j > lo
            if Base.lt(o, d, distance²(p, v[j-1]))
                v[j] = v[j-1]
                j -= 1
                continue
            end
            break
        end
        v[j] = pt
    end
    return v
end

@inline function selectpivot!(p::Point, 
                              v::Vector{<:Point}, 
                              lo::Integer, 
                              hi::Integer,
                              o::Base.Ordering)
    @inbounds begin
        mi = Base.Sort.midpoint(lo, hi)

        # sort v[mi] <= v[lo] <= v[hi] such that the pivot is immediately in place
        if Base.lt(o, distance²(p, v[lo]), distance²(p, v[mi]))
            v[mi], v[lo] = v[lo], v[mi]
        end

        if Base.lt(o, distance²(p, v[hi]), distance²(p, v[lo]))
            if Base.lt(o, distance²(p, v[hi]), distance²(p, v[mi]))
                v[hi], v[lo], v[mi] = v[lo], v[mi], v[hi]
            else
                v[hi], v[lo] = v[lo], v[hi]
            end
        end

        # return the pivot
        return v[lo]
    end
end

function partition!(p::Point, v::Vector{<:Point}, lo::Integer, hi::Integer, o::Base.Ordering)
    pivot = selectpivot!(p, v, lo, hi, o)
    d = distance²(p, pivot)
    # pivot == v[lo], v[hi] > pivot
    i, j = lo, hi
    @inbounds while true
        i += 1; j -= 1
        while Base.lt(o, distance²(p, v[i]), d); i += 1; end;
        while Base.lt(o, d, distance²(p, v[j])); j -= 1; end;
        i >= j && break
        v[i], v[j] = v[j], v[i]
    end
    v[j], v[lo] = pivot, v[j]

    # v[j] == pivot
    # v[k] >= pivot for k > j
    # v[i] <= pivot for i < j
    return j
end

function Base.sort!(p::Point, 
                    v::Vector{<:Point}, 
                    lo::Integer, 
                    hi::Integer,
                    a::Base.Sort.QuickSortAlg, 
                    o::Base.Ordering)
    @inbounds while lo < hi
        if hi-lo ≤ Base.Sort.SMALL_THRESHOLD
            return sort!(p, v, lo, hi, Base.Sort.InsertionSort, o)
        end
        j = partition!(p, v, lo, hi, o)
        if j-lo < hi-j
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            lo < (j-1) && sort!(p, v, lo, j-1, a, o)
            lo = j+1
        else
            j+1 < hi && sort!(p, v, j+1, hi, a, o)
            hi = j-1
        end
    end
    return v
end
