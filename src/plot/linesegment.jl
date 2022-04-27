function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
    return convert_arguments(LS, [l[1], l[2]])
end

function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment})
    return convert_arguments(LS, reduce(vcat, [[l[1], l[2]] for l in L]))
end
