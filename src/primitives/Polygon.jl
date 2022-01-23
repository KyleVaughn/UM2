# A polygon defined by its vertices in counterclockwise order 
struct Polygon{N,Dim,T} <: Face{Dim,1,T}
    points::SVector{N, Point{Dim,T}}
end

# Aliases for convenience
const Triangle      = Polygon{3}
const Quadrilateral = Polygon{4}
const Pentagon      = Polygon{5}
const Hexagon       = Polygon{6}
const Heptagon      = Polygon{7}
const Octagon       = Polygon{8}
const Nonagon       = Polygon{9}
const Decagon       = Polygon{10}

Base.@propagate_inbounds function Base.getindex(poly::Polygon, i::Integer)
    getfield(poly, :points)[i]
end

# Constructors
# ---------------------------------------------------------------------------------------------
Polygon{N,Dim,T}(x...) where {N,Dim,T} = Polygon{N,Dim,T}(SVector{N, Point{Dim,T}}(x))
Polygon{Dim,T}(x...) where {Dim,T} = Polygon(SVector(x))
Polygon{Dim}(x...) where {Dim} = Polygon(SVector(x))
Polygon(x...) = Polygon(SVector(x))

# Methods
# ---------------------------------------------------------------------------------------------
# Shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)
function area(poly::Polygon{N,Dim,T}) where {N,Dim,T}
    # This can be done with mapreduce, but mapreduce is substantially slower
    # Set a equal to the scalar or vector 0
    a = Base.zero(Point{Dim,T}) × Base.zero(Point{Dim,T}) 
    for i = 1:N-1
        a += poly[i] × poly[i+1]
    end
    a += poly[N] × poly[1]
    return a/2
end
# We can go faster on triangles with the simplification below
function area(tri::Triangle)
    𝘂 = tri[2] - tri[1]
    𝘃 = tri[3] - tri[1]
    return (𝘂 × 𝘃)/2
end
# centroid(tri::Triangle) = tri(1//3, 1//3)
# 
# function Base.in(p::Point_2D, tri::Triangle_2D)
#     # If the point is to the left of every edge
#     #  3<-----2
#     #  |     ^
#     #  | p  /
#     #  |   /
#     #  |  /
#     #  v /
#     #  1
#     return isleft(p, LineSegment_2D(tri[1], tri[2])) &&
#            isleft(p, LineSegment_2D(tri[2], tri[3])) &&
#            isleft(p, LineSegment_2D(tri[3], tri[1]))
# end
# 
# function Base.intersect(l::LineSegment_2D{T}, tri::Triangle_2D{T}) where {T}
#     # Create the 3 line segments that make up the triangle and intersect each one
#     p₁ = Point_2D{T}(0,0)
#     p₂ = Point_2D{T}(0,0)
#     p₃ = Point_2D{T}(0,0)
#     npoints = 0x0000
#     for i ∈ 1:3
#         hit, point = l ∩ LineSegment_2D(tri[(i - 1) % 3 + 1], 
#                                         tri[      i % 3 + 1])
#         if hit
#             npoints += 0x0001
#             if npoints === 0x0001
#                 p₁ = point
#             elseif npoints === 0x0002
#                 p₂ = point
#             else
#                 p₃ = point
#             end
#         end
#     end
#     return npoints, SVector(p₁, p₂, p₃) 
# end
# 
# # Plot
# # ---------------------------------------------------------------------------------------------
# if enable_visualization
#     function convert_arguments(LS::Type{<:LineSegments}, tri::Triangle)
#         l₁ = LineSegment(tri[1], tri[2])
#         l₂ = LineSegment(tri[2], tri[3])
#         l₃ = LineSegment(tri[3], tri[1])
#         lines = [l₁, l₂, l₃]
#         return convert_arguments(LS, lines)
#     end
# 
#     function convert_arguments(LS::Type{<:LineSegments}, T::Vector{<:Triangle})
#         point_sets = [convert_arguments(LS, tri) for tri ∈  T]
#         return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
#     end
# 
#     function convert_arguments(M::Type{<:Mesh}, tri::Triangle)
#         points = [tri[i].coord for i = 1:3]
#         face = [1 2 3]
#         return convert_arguments(M, points, face)
#     end
# 
#     function convert_arguments(M::Type{<:Mesh}, T::Vector{<:Triangle})
#         points = reduce(vcat, [[tri[i].coord for i = 1:3] for tri ∈  T])
#         faces = zeros(Int64, length(T), 3)
#         k = 1
#         for i in 1:length(T), j = 1:3
#             faces[i, j] = k
#             k += 1
#         end
#         return convert_arguments(M, points, faces)
#     end
# end
