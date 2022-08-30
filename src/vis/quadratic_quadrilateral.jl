function convert_arguments(LS::Type{<:LineSegments}, quad::QuadraticQuadrilateral)
    return convert_arguments(LS, collect(edge_iterator(quad)))
end

function convert_arguments(LS::Type{<:LineSegments}, 
                           quads::Vector{QuadraticQuadrilateral{D, T}}) where {D, T}
    edges = Vector{QuadraticSegment{D, T}}(undef, 4 * length(quads))
    for i in eachindex(quads)
        edges[4 * i - 3] = edge(1, quads[i])
        edges[4 * i - 2] = edge(2, quads[i])
        edges[4 * i - 1] = edge(3, quads[i])
        edges[4 * i    ] = edge(4, quads[i])
    end
    return convert_arguments(LS, edges) 
end

function convert_arguments(M::Type{<:GLMakieMesh}, quad::QuadraticQuadrilateral)
    NDIV = UM2_VIS_NONLINEAR_SUBDIVISIONS
    return convert_arguments(M, triangulate(quad, NDIV)) 
end

function convert_arguments(M::Type{<:GLMakieMesh}, 
                           quads::Vector{QuadraticQuadrilateral{D, T}}) where {D, T}
    NDIV = UM2_VIS_NONLINEAR_SUBDIVISIONS
    nquad = 2 * NDIV * NDIV
    triangles = Vector{Triangle{D, T}}(undef, nquad * length(quads))
    for i in eachindex(quads)
        tris = triangulate(quads[i], NDIV)
        for j in 1:nquad
            triangles[nquad * (i - 1) + j] = tris[j]
        end
    end
    return convert_arguments(M, triangles)
end
