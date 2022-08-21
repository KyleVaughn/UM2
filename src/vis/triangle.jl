function convert_arguments(LS::Type{<:LineSegments}, tri::Triangle)
    return convert_arguments(LS, collect(edges(tri)))
end

function convert_arguments(LS::Type{<:LineSegments}, 
                           tris::Vector{Triangle{D, T}}) where {D, T}
    lines = Vector{LineSegment{D, T}}(undef, 3 * length(tris))
    for i in eachindex(tris)
        lines[3 * i - 2] = edge(1, tris[i])
        lines[3 * i - 1] = edge(2, tris[i])
        lines[3 * i    ] = edge(3, tris[i])
    end
    return convert_arguments(LS, lines)
end

function convert_arguments(M::Type{<:GLMakieMesh}, tri::Triangle)
    verts = map(coord, vertices(tri))
    face = [1 2 3]
    return convert_arguments(M, verts, face)
end

function convert_arguments(
        M::Type{<:GLMakieMesh},
        tris::Vector{Triangle{D, T}}) where {D, T}
    points = Vector{NTuple{D, T}}(undef, 3 * length(tris))
    for i in eachindex(tris)
        tri            = tris[i]
        points[3i - 2] = coord(tri[1])
        points[3i - 1] = coord(tri[2])
        points[3i]     = coord(tri[3])
    end
    faces = zeros(Int64, length(tris), 3)
    k = 1
    for i in eachindex(tris), j in 1:3
        faces[i, j] = k
        k += 1
    end
    return convert_arguments(M, points, faces)
end
