function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
    return convert_arguments(LS, vertices(l))
end

function convert_arguments(
        LS::Type{<:LineSegments},
        lines::Vector{LineSegment{D, T}}) where {D, T}
    points = Vector{NTuple{D, T}}(undef, 2 * length(lines))
    for i in eachindex(lines)
        points[2 * i - 1] = coord(lines[i][1])
        points[2 * i    ] = coord(lines[i][2])
    end
    return convert_arguments(LS, points)
end
