# Sort points based on their distance from another point.
# Default algorithm is quicksort. If the vector is less than 20 elements, insertion sort
# is used instead.
# This is essentially Julia's Base.Sort

defalg(v::Vector{<:Point}) = Base.Sort.QuickSort
function Base.sort!(p::Point,
                    v::Vector{<:Point};
                    alg::Base.Sort.Algorithm = defalg(v),
                    order::Base.Ordering = Base.Forward)
    return sort!(p, v, firstindex(v), lastindex(v), alg, order)
end

function Base.sort(p::Point,
                   v::Vector{<:Point};
                   alg::Base.Sort.Algorithm = defalg(v),
                   order::Base.Ordering = Base.Forward)
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
    @inbounds for i in (lo + 1):hi
        j = i
        d = distance²(p, v[i])
        pt = v[i]
        while j > lo
            if Base.lt(o, d, distance²(p, v[j - 1]))
                v[j] = v[j - 1]
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

function partition!(p::Point, v::Vector{<:Point}, lo::Integer, hi::Integer,
                    o::Base.Ordering)
    pivot = selectpivot!(p, v, lo, hi, o)
    d = distance²(p, pivot)
    # pivot == v[lo], v[hi] > pivot
    i, j = lo, hi
    @inbounds while true
        i += 1
        j -= 1
        while Base.lt(o, distance²(p, v[i]), d)
            i += 1
        end
        while Base.lt(o, d, distance²(p, v[j]))
            j -= 1
        end
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
        if hi - lo ≤ Base.Sort.SMALL_THRESHOLD
            return sort!(p, v, lo, hi, Base.Sort.InsertionSort, o)
        end
        j = partition!(p, v, lo, hi, o)
        if j - lo < hi - j
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            lo < (j - 1) && sort!(p, v, lo, j - 1, a, o)
            lo = j + 1
        else
            j + 1 < hi && sort!(p, v, j + 1, hi, a, o)
            hi = j - 1
        end
    end
    return v
end

# p is the reference point, x is being found in v
function findsortedfirst(p::Point, v::Vector{<:Point}, x::Point)
    d = distance²(p, x)
    for i in eachindex(v)
        d ≤ distance²(p, v[i]) && return i
    end
    return length(v) + 1
end

function Base.searchsortedfirst(p::Point, v::Vector{<:Point}, x::Point,
                                lo::T, hi::T,
                                o::Base.Ordering)::keytype(v) where {T <: Integer}
    d = distance²(p, x)
    u = T(1)
    lo = lo - u
    hi = hi + u
    @inbounds while lo < hi - u
        m = Base.Sort.midpoint(lo, hi)
        if Base.lt(o, distance²(p, v[m]), d)
            lo = m
        else
            hi = m
        end
    end
    return hi
end

function getsortedfirst(p::Point, v::Vector{<:Point}, x::Point)
    if SORTED_ARRAY_THRESHOLD ≤ length(v)
        return searchsortedfirst(p, v, x, firstindex(v), lastindex(v), Base.Forward)
    else
        return findsortedfirst(p, v, x)
    end
end
