function convert_arguments(LS::Type{<:LineSegments}, r::Ray)
    return convert_arguments(LS, [r.origin, r. origin + r.direction])
end

function convert_arguments(LS::Type{<:LineSegments},rays::Vector{Ray{D, T}}) where {D, T}
    points = Vector{NTuple{D, T}}(undef, 2*length(rays))
    for i in eachindex(rays)
        points[2 * i - 1] = coord(rays[i].origin)
        points[2 * i    ] = coord(rays[i].origin + rays[i].direction)
    end
    return convert_arguments(LS, points)
end
