# If a sorted array has length less than this value, findfirst will be the
# quickest method to get the desired index. Otherwise, searchsortedfirst will be
# the quickest.
const sorted_array_findfirst_threshold = 60 

function findsortedfirst(v::AbstractVector, x)
    for i in eachindex(v)
        x ≤ v[i] && return i
    end 
    return length(v) + 1 
end

function getsortedfirst(v::AbstractVector, x)
    if sorted_array_findfirst_threshold ≤ length(v)
        return searchsortedfirst(v, x)
    else
        return findsortedfirst(v, x)
    end
end
