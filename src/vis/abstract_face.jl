function convert_arguments(LS::Type{<:LineSegments}, face::AbstractFace)
    return convert_arguments(LS, collect(edges(face)))
end

function convert_arguments(LS::Type{<:LineSegments}, faces::Vector{<:AbstractFace})
    return convert_arguments(
                LS, 
                vcat([convert_arguments(LS, face)[1] for face in faces]...)
               )
end

function convert_arguments(M::Type{<:GLMakieMesh}, tri::Triangle)
    verts = [coord(v) for v in vertices(tri)]
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
    for i in 1:length(tris), j in 1:3
        faces[i, j] = k
        k += 1
    end
    return convert_arguments(M, points, faces)
end

function convert_arguments(M::Type{<:GLMakieMesh}, poly::AbstractLinearPolygon)
    return convert_arguments(M, collect(triangulate(poly)))
end

function convert_arguments(M::Type{<:GLMakieMesh}, poly::AbstractQuadraticPolygon)
    NDIV = UM2_VIS_NONLINEAR_SUBDIVISIONS
    return convert_arguments(M, triangulate(poly, NDIV)) 
end

function convert_arguments(M::Type{<:GLMakieMesh}, P::Vector{<:AbstractPolygon})
    triangles = [begin if poly isa Triangle 
                    [poly]
                 elseif poly isa AbstractLinearPolygon
                     triangulate(poly)
                 else
                     triangulate(poly, UM2_VIS_NONLINEAR_SUBDIVISIONS) 
                 end end
                 for poly in P]
    return convert_arguments(M, reduce(vcat, triangles))
end
