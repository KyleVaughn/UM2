function convert_arguments(LS::Type{<:LineSegments}, q::Quadrilateral)
    return convert_arguments(LS, collect(edges(q)))
end

function convert_arguments(LS::Type{<:LineSegments}, 
                           quads::Vector{Quadrilateral{D, T}}) where {D, T}
    lines = Vector{LineSegment{D, T}}(undef, 4 * length(quads))    
    for i in eachindex(quads)    
        lines[4 * i - 3] = edge(1, quads[i])
        lines[4 * i - 2] = edge(2, quads[i])
        lines[4 * i - 1] = edge(3, quads[i])
        lines[4 * i    ] = edge(4, quads[i])
    end    
    return convert_arguments(LS, lines)    
end

function convert_arguments(M::Type{<:GLMakieMesh}, q::Quadrilateral)
    return convert_arguments(M, collect(triangulate(q)))
end

function convert_arguments(M::Type{<:GLMakieMesh}, quads::Vector{<:Quadrilateral})
    triangles = Vector{Triangle{D, T}}(undef, 2 * length(quads))
    for i in eachindex(quads)
        tris = triangulate(quads[i])
        triangles[2 * i - 1] = tris[1]
        triangles[2 * i    ] = tris[2]
    end
    return convert_arguments(M, triangles)
end
