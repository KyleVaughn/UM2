function convert_arguments(LS::Type{<:LineSegments}, aab::AABox{2})
    return convert_arguments(LS, facets(aab))
end

function convert_arguments(LS::Type{<:LineSegments}, aab::AABox{3})
    return convert_arguments(LS, ridges(aab))
end

function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABox})
    point_sets = [convert_arguments(LS, aab) for aab in R]
    return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
end

function convert_arguments(LS::Type{<:LineSegments}, R::SVector{N, <:AABox}) where {N}
    return convert_arguments(LS, collect(R))
end

function convert_arguments(M::Type{<:GLMakieMesh}, aab::AABox{2})
    vertices = [v.coords for v in ridges(aab)]
    faces = [1 2 3;
             3 4 1]
    return convert_arguments(M, vertices, faces)
end

function convert_arguments(M::Type{<:GLMakieMesh}, aab::AABox{3})
    return convert_arguments(M, facets(aab)) 
end
