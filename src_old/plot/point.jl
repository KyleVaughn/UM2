function convert_arguments(T::Type{<:Scatter}, p::Point)
    return convert_arguments(T, coordinates(p))
end

function convert_arguments(T::Type{<:Scatter}, P::Vector{<:Point})
    return convert_arguments(T, [coordinates(p) for p in P])
end

function convert_arguments(T::Type{<:LineSegments}, p::Point)
    return convert_arguments(T, coordinates(p))
end

function convert_arguments(T::Type{<:LineSegments}, P::Vector{<:Point})
    return convert_arguments(T, [coordinates(p) for p in P])
end

# Makie seems to have trouble with SVectors, so just collect them into Vectors
function convert_arguments(T::Type{<:Scatter}, V::SVector{N, <:Point}) where {N}
    return convert_arguments(T, collect(V))
end

function convert_arguments(T::Type{<:LineSegments}, V::SVector{N, <:Point}) where {N}
    return convert_arguments(T, collect(V))
end
