function convert_arguments(LS::Type{<:LineSegments}, aab::AABox{2})
    return convert_arguments(LS, edges(aab))
end

function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABox})
    return convert_arguments(LS, vcat([convert_arguments(LS, aab)[1] for aab ∈ R]...))
end

function convert_arguments(LS::Type{<:LineSegments}, R::SVector{N, <:AABox}) where {N}
    return convert_arguments(LS, collect(R))
end

function convert_arguments(M::Type{<:GLMakieMesh}, aab::AABox{2})
    vertices = [coordinates(v) for v in vertices(aab)]
    faces = [1 2 3;
             3 4 1]
    return convert_arguments(M, vertices, faces)
end

function convert_arguments(M::Type{<:GLMakieMesh}, aab::AABox{3})
    return convert_arguments(M, faces(aab)) 
end
