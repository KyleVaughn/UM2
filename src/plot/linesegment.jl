function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
    return convert_arguments(LS, collect(vertices(l)))
end

function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment})
    return convert_arguments(LS, vcat([[l[1], l[2]] for l in L]...))
end
