function convert_arguments(T::Type{<:Scatter}, p::Point)
    return convert_arguments(T, p.coords)
end 

function convert_arguments(T::Type{<:Scatter}, P::Vector{<:Point})
    return convert_arguments(T, [p.coords for p in P]) 
end 

function convert_arguments(T::Type{<:LineSegments}, p::Point)
    return convert_arguments(T, p.coords)
end 

function convert_arguments(T::Type{<:LineSegments}, P::Vector{<:Point})
    return convert_arguments(T, [p.coords for p in P]) 
end
