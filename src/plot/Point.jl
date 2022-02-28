function convert_arguments(T::Type{<:Scatter}, p::Point)
    return convert_arguments(T, p.coord)
end 

function convert_arguments(T::Type{<:Scatter}, P::Vector{<:Point})
    return convert_arguments(T, [p.coord for p in P]) 
end 
