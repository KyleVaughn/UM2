function convert_arguments(LS::Type{<:LineSegments}, poly::QuadraticPolygon{N}) where {N}
    M = N ÷ 2
    qsegs = [QuadraticSegment(poly[(i - 1) % M + 1],  
                              poly[      i % M + 1],
                              poly[          i + M]) for i = 1:M]
    return convert_arguments(LS, qsegs)
end

function convert_arguments(LS::Type{<:LineSegments}, P::Vector{<:QuadraticPolygon})
    point_sets = [convert_arguments(LS, poly) for poly ∈ P]
    return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
end

function convert_arguments(P::Type{<:Mesh}, poly::QuadraticPolygon)
    triangles = collect(triangulate(poly, Val(plot_nonlinear_subdivisions)))
    return convert_arguments(P, triangles)
end

function convert_arguments(M::Type{<:Mesh}, P::Vector{<:QuadraticPolygon})
    triangles = reduce(vcat, 
                       [collect(
                                triangulate(p, Val(plot_nonlinear_subdivisions)
                                           )
                               ) for p in P]
                      )
    return convert_arguments(M, triangles)
end
