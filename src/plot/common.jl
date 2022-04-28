# Makie seems to have trouble with SVectors, so just collect them into Vectors
function convert_arguments(T::Type{<:LineSegments}, 
                           V::SVector{NP, Polytope{K,P,N,PT}}) where {NP,K,P,N,PT}
    return convert_arguments(T, collect(V)) 
end

function convert_arguments(T::Type{<:Mesh}, 
                           V::SVector{NP, Polytope{K,P,N,PT}}) where {NP,K,P,N,PT}
    return convert_arguments(T, collect(V)) 
end
