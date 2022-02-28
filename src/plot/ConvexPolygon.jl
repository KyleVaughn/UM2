function convert_arguments(LS::Type{<:LineSegments}, poly::ConvexPolygon{N}
                          ) where {N}
    lines = [LineSegment(poly[(i-1) % N + 1],
                         poly[    i % N + 1]) for i = 1:N]
    return convert_arguments(LS, lines)
end

function convert_arguments(LS::Type{<:LineSegments}, P::Vector{<:ConvexPolygon})
    point_sets = [convert_arguments(LS, poly) for poly ∈  P]
    return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
end

function convert_arguments(M::Type{<:Mesh}, tri::Triangle)
    points = [tri[i].coord for i = 1:3]
    face = [1 2 3]
    return convert_arguments(M, points, face)
end

function convert_arguments(M::Type{<:Mesh}, T::Vector{<:Triangle})
    points = reduce(vcat, [[tri[i].coord for i = 1:3] for tri ∈  T])
    faces = zeros(Int64, length(T), 3)
    k = 1
    for i in 1:length(T), j = 1:3
        faces[i, j] = k
        k += 1
    end
    return convert_arguments(M, points, faces)
end

function convert_arguments(M::Type{<:Mesh}, quad::Quadrilateral)
    return convert_arguments(M, collect(triangulate(quad, Val(0))))
end

function convert_arguments(M::Type{<:Mesh}, P::Vector{<:Quadrilateral})
    return convert_arguments(M, reduce(vcat, [collect(triangulate(quad, Val(0))) for quad in P]))
end
