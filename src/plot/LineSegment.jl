function convert_arguments(T::Type{<:LineSegments}, p::Point)
    return convert_arguments(T, p.coord)
end 

function convert_arguments(T::Type{<:LineSegments}, P::Vector{<:Point})
    return convert_arguments(T, [p.coord for p in P]) 
end

function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
    return convert_arguments(LS, [l.𝘅₁, l.𝘅₂])
end

function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment})
    return convert_arguments(LS, reduce(vcat, [[l.𝘅₁, l.𝘅₂] for l in L]))
end
