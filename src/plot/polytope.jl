# LineSegments
function convert_arguments(LS::Type{<:LineSegments}, poly::Polytope)
    return convert_arguments(LS, edges(poly))
end

function convert_arguments(LS::Type{<:LineSegments}, P::Vector{<:Polytope})
    point_sets = [convert_arguments(LS, poly) for poly ∈  P]
    return convert_arguments(LS, vcat([pset[1] for pset ∈ point_sets]...))
end

# Mesh 
# Triangulate
function convert_arguments(M::Type{<:GLMakieMesh}, tri::Triangle)
    vertices = [v.coords for v in ridges(tri)]
    face = [1 2 3]
    return convert_arguments(M, vertices, face)
end

function convert_arguments(M::Type{<:GLMakieMesh}, T::Vector{<:Triangle})
    points = reduce(vcat, [[coordinates(v) for v ∈ vertices(tri)] for tri ∈  T]) 
    faces = zeros(Int64, length(T), 3)
    k = 1
    for i in 1:length(T), j = 1:3
        faces[i, j] = k
        k += 1
    end
    return convert_arguments(M, points, faces)
end

# 2D Polygons can be triangulated using fan triangulation.
# All others need to be triangulated based on the number of non-linear subdivisions
function convert_arguments(M::Type{<:GLMakieMesh}, poly::Polygon{N, Point{2,T}}) where {N,T}
    return convert_arguments(M, triangulate(poly))
end

function convert_arguments(M::Type{<:GLMakieMesh}, poly::Polytope{2})
    return convert_arguments(M, triangulate(poly, Val(plot_nonlinear_subdivisions)))
end
 
function convert_arguments(M::Type{<:GLMakieMesh}, P::Vector{<:Polytope{2}})
    triangles = [ begin
                     if poly isa Polygon{N, Point{2,T}} where {N,T}
                         triangulate(poly)
                     elseif poly isa Triangle
                         Vec(poly)
                     else
                         triangulate(poly, Val(plot_nonlinear_subdivisions))
                     end
                  end
                  for poly in P]
    return convert_arguments(M, reduce(vcat, triangles))
end

function convert_arguments(M::Type{<:GLMakieMesh}, poly::Polytope{3})
    return convert_arguments(M, faces(poly)) 
end
