function convert_arguments(LS::Type{<:LineSegments}, tri::QuadraticTriangle)
    return convert_arguments(LS, collect(edges(tri)))
end

function convert_arguments(LS::Type{<:LineSegments}, 
                           tris::Vector{QuadraticTriangle{D, T}}) where {D, T}
    edges = Vector{QuadraticSegment{D, T}}(undef, 3 * length(tris))
    for i in eachindex(tris)
        edges[3 * i - 2] = edge(1, tris[i])
        edges[3 * i - 1] = edge(2, tris[i])
        edges[3 * i    ] = edge(3, tris[i])
    end
    return convert_arguments(LS, edges) 
end

function convert_arguments(M::Type{<:GLMakieMesh}, tri::QuadraticTriangle)
    NDIV = UM2_VIS_NONLINEAR_SUBDIVISIONS
    return convert_arguments(M, triangulate(tri, NDIV)) 
end

function convert_arguments(M::Type{<:GLMakieMesh}, 
                           qtris::Vector{QuadraticTriangle{D, T}}) where {D, T}
    NDIV = UM2_VIS_NONLINEAR_SUBDIVISIONS
    ntri = NDIV * NDIV
    triangles = Vector{Triangle{D, T}}(undef, ntri * length(qtris))
    for i in eachindex(qtris)
        tris = triangulate(qtris[i], NDIV)
        for j in 1:ntri
            triangles[ntri * (i - 1) + j] = tris[j]
        end
    end
    return convert_arguments(M, triangles)
end
