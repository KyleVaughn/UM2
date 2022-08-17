function convert_arguments(T::Type{<:Scatter}, p::Point)
    return convert_arguments(T, p.coord)
end

function convert_arguments(T::Type{<:Scatter}, points::Vector{<:Point})
    return convert_arguments(T, map(coord, points))
end

function convert_arguments(T::Type{<:LineSegments}, 
        points::Vector{Point{D, F}}) where {D, F}
    N = length(points)
    points_doubled = Vector{NTuple{D, F}}(undef, 2*(N - 1))
    points_doubled[1] = coord(points[1])
    for i = 2:(N-1)
        points_doubled[2*(i-1)    ] = coord(points[i])
        points_doubled[2*(i-1) + 1] = coord(points[i])
    end
    points_doubled[2*(N-1)] = coord(points[N])
    return convert_arguments(T, points_doubled)
end

function convert_arguments(T::Type{<:Scatter}, points::Vec{N, <:Point}) where {N}
    return convert_arguments(T, collect(points))
end

function convert_arguments(T::Type{<:LineSegments}, points::Vec{N, <:Point}) where {N}
    return convert_arguments(T, collect(points))
end
