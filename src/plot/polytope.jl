# Makie seems to have trouble with SVectors, so just collect them into Vectors
function convert_arguments(T::Type{<:LineSegments},
                           V::SVector{NP, Polytope{K, P, N, PT}}) where {NP, K, P, N, PT}
    return convert_arguments(T, collect(V))
end

function convert_arguments(T::Type{<:GLMakieMesh},
                           V::SVector{NP, Polytope{K, P, N, PT}}) where {NP, K, P, N, PT}
    return convert_arguments(T, collect(V))
end

# LineSegment
function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment)
    return convert_arguments(LS, collect(vertices(l)))
end

function convert_arguments(LS::Type{<:LineSegments},
                           L::Vector{LineSegment{T}}) where {T}
    points = Vector{T}(undef, 2length(L))
    for i in eachindex(L)
        points[2i - 1] = L[i][1]
        points[2i]     = L[i][2]
    end
    return convert_arguments(LS, points)
end

# QuadraticSegment
function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment{T}) where {T}
    points = Vector{T}(undef, 2 * (plot_nonlinear_subdivisions - 1))
    r = LinRange(0, 1, plot_nonlinear_subdivisions)
    for i in 1:(plot_nonlinear_subdivisions - 1)
        points[2i - 1] = q(r[i])
        points[2i]     = q(r[i + 1])
    end
    return convert_arguments(LS, points)
end

function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:QuadraticSegment})
    return convert_arguments(LS, vcat([convert_arguments(LS, q)[1] for q in Q]...))
end

# General Polytopes
#
# LineSegments
function convert_arguments(LS::Type{<:LineSegments}, poly::Polytope)
    return convert_arguments(LS, edges(poly))
end

function convert_arguments(LS::Type{<:LineSegments}, P::Vector{<:Polytope})
    return convert_arguments(LS, vcat([convert_arguments(LS, p)[1] for p in P]...))
end

# Mesh 
# 2D Polygons can be triangulated using fan triangulation.
# All others need to be triangulated based on the number of nonlinear subdivisions
function convert_arguments(M::Type{<:GLMakieMesh}, tri::Triangle)
    vertices = [coordinates(v) for v in ridges(tri)]
    face = [1 2 3]
    return convert_arguments(M, vertices, face)
end

function convert_arguments(M::Type{<:GLMakieMesh},
                           tris::Vector{Triangle{Point{Dim, T}}}) where {Dim, T}
    points = Vector{Vec{Dim, T}}(undef, 3length(tris))
    for i in eachindex(tris)
        tri            = tris[i]
        points[3i - 2] = coordinates(tri[1])
        points[3i - 1] = coordinates(tri[2])
        points[3i]     = coordinates(tri[3])
    end
    faces = zeros(Int64, length(tris), 3)
    k = 1
    for i in 1:length(tris), j in 1:3
        faces[i, j] = k
        k += 1
    end
    return convert_arguments(M, points, faces)
end

function convert_arguments(M::Type{<:GLMakieMesh},
                           poly::Polygon{N, Point{2, T}}) where {N, T}
    return convert_arguments(M, triangulate(poly))
end

function convert_arguments(M::Type{<:GLMakieMesh}, poly::Polytope{2})
    return convert_arguments(M, triangulate(poly, Val(plot_nonlinear_subdivisions)))
end

function convert_arguments(M::Type{<:GLMakieMesh}, P::Vector{<:Polytope{2}})
    triangles = [begin if poly isa Polygon{N, Point{2, T}} where {N, T}
                     triangulate(poly)
                 elseif poly isa Triangle
                     Vec(poly)
                 else
                     triangulate(poly, Val(plot_nonlinear_subdivisions))
                 end end
                 for poly in P]
    return convert_arguments(M, reduce(vcat, triangles))
end

function convert_arguments(M::Type{<:GLMakieMesh}, poly::Polytope{3})
    return convert_arguments(M, faces(poly))
end
