# A triangle with quadratic edges.
struct Triangle6{N,T} <: Face{N,T}
    # The points are assumed to be ordered as follows
    # p₁ = vertex A
    # p₂ = vertex B
    # p₃ = vertex C
    # p₄ = point on the quadratic segment from A to B
    # p₅ = point on the quadratic segment from B to C
    # p₆ = point on the quadratic segment from C to A
    points::SVector{6, Point{N,T}}
end

const Triangle6_2D = Triangle6{2}
const Triangle6_3D = Triangle6{3}

Base.@propagate_inbounds function Base.getindex(tri6::Triangle6, i::Integer)
    getfield(tri6, :points)[i]
end

# Constructors
# ---------------------------------------------------------------------------------------------
function Triangle6(p₁::Point{N,T}, p₂::Point{N,T}, p₃::Point{N,T},
                   p₄::Point{N,T}, p₅::Point{N,T}, p₆::Point{N,T}) where {N,T}
    return Triangle6{N,T}(SVector{6, Point{N,T}}(p₁, p₂, p₃, p₄, p₅, p₆))
end
function Triangle6{N}(p₁::Point{N,T}, p₂::Point{N,T}, p₃::Point{N,T},
                      p₄::Point{N,T}, p₅::Point{N,T}, p₆::Point{N,T}) where {N,T}
    return Triangle6{N,T}(SVector{6, Point{N,T}}(p₁, p₂, p₃, p₄, p₅, p₆))
end

# Methods
# ---------------------------------------------------------------------------------------------
# Interpolation
function (tri6::Triangle6)(r, s)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    return Point((1 - r - s)*(2(1 - r - s) - 1)*tri6[1] +
                                     r*(2r - 1)*tri6[2] +
                                     s*(2s - 1)*tri6[3] +
                                 4r*(1 - r - s)*tri6[4] +
                                         (4r*s)*tri6[5] +
                                 4s*(1 - r - s)*tri6[6] )
end

function (tri6::Triangle6)(p::Point_2D)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    r = p[1]; s = p[2]
    return Point((1 - r - s)*(2(1 - r - s) - 1)*tri6[1] +
                                     r*(2r - 1)*tri6[2] +
                                     s*(2s - 1)*tri6[3] +
                                 4r*(1 - r - s)*tri6[4] +
                                         (4r*s)*tri6[5] +
                                 4s*(1 - r - s)*tri6[6] )
end


function area(tri6::Triangle6_2D)
    # Let F(r,s) be the interpolation function for tri6, and 𝗝(r,s) be the Jacobian of F 
    # at (r,s). |𝗝| is the Jacobian determinant 
    #                    1 1-r                N
    # A = ∬ |𝗝(r,s)|dA = ∫  ∫ |𝗝(r,s)|ds dr = ∑ wᵢ|𝗝(rᵢ,sᵢ)|
    #     D              0  0                i=1
    # Mathematica for this algebraic nightmare
    a = tri6[1][1]; g = tri6[1][2]
    b = tri6[2][1]; h = tri6[2][2]
    c = tri6[3][1]; i = tri6[3][2]
    d = tri6[4][1]; j = tri6[4][2]
    e = tri6[5][1]; k = tri6[5][2]
    f = tri6[6][1]; l = tri6[6][2]
    return (-4d*g + 4f*g -a*h + 4d*h - 4e*h + a*i + 4e*i - 4f*i + 
            4a*j + b*(g-i-4j+4k) - 4a*l + c*(-g+h-4k+4l))/6
end


#USE VIEWS FOR SLICE
# Determine 3D Val(N) in the future. to resurrect this method. For now, leave it be. 
# area(tri6::Triangle6_2D) = area(tri6, Val(3))
# function area(tri6::Triangle6, ::Val{N}) where {N}
#     # Gauss-Legendre quadrature over a triangle is used.
#     # Let F(r,s) be the interpolation function for tri6, and 𝗝(r,s) be the Jacobian of F 
#     # at (r,s). |𝗝| is the Jacobian determinant 
#     #                           1 1-r                       N
#     # A = ∬ ‖∂F/∂r × ∂F/∂s‖dA = ∫  ∫ ‖∂F/∂r × ∂F/∂s‖ds dr = ∑ wᵢ‖∂F/∂r(rᵢ,sᵢ) × ∂F/∂s(rᵢ, sᵢ)‖
#     #     D                     0  0                       i=1
#     #
#     # N is the number of points used in the quadrature.
#     w, r, s = gauss_legendre_quadrature(tri6, Val(N))
#     return sum(@. w * norm( 𝗝(tri6, r, s) |> x->x[:,1] × x[:,2]))
# end

centroid(tri6::Triangle6_2D) = centroid(tri6, Val(6))
function centroid(tri6::Triangle6_2D, ::Val{N}) where {N}
    # Gauss-Legendre quadrature over a triangle is used.
    # Let F(r,s) be the interpolation function for tri6,
    #                    1 1-r                N                
    # A = ∬ |𝗝(r,s)|dA = ∫  ∫ |𝗝(r,s)|ds dr = ∑ wᵢ|𝗝(rᵢ,sᵢ)|
    #     D              0  0                i=1
    #
    # C_x = (∫∫ x dA)/A, C_y = (∫∫ y dA)/A
    #         D                  D
    w, r, s = gauss_legendre_quadrature(tri6, Val(N))
    # We can reuse our computed weighted Jacobian determinants, since we need these
    # in the C_y, C_y, and A.
    weighted_vals = @. w * det(𝗝(tri6, r, s))
    points = tri6.(r, s)
    A = sum(weighted_vals)
    C_x = sum(getindex.(points, 1) .* weighted_vals)
    C_y = sum(getindex.(points, 2) .* weighted_vals)
    return Point_2D(C_x/A, C_y/A)
end

function jacobian(tri6::Triangle6, r, s)
    # Let F(r,s) be the interpolation function for tri6
    # Returns ∂F/∂r, ∂F/∂s
    ∂F_∂r = (4r + 4s - 3)*tri6[1] +
                 (4r - 1)*tri6[2] +
            4(1 - 2r - s)*tri6[4] +
                     (4s)*tri6[5] +
                    (-4s)*tri6[6]

    ∂F_∂s = (4r + 4s - 3)*tri6[1] +
                 (4s - 1)*tri6[3] +
                    (-4r)*tri6[4] +
                     (4r)*tri6[5] +
            4(1 - r - 2s)*tri6[6]
    return hcat(∂F_∂r, ∂F_∂s)
end

function Base.in(p::Point_2D, tri6::Triangle6_2D)
    # If the point is to the left of every edge
    #  3<-----2
    #  |     ^
    #  | p  /
    #  |   /
    #  |  /
    #  v /
    #  1
    return isleft(p, QuadraticSegment_2D(tri6[1], tri6[2], tri6[4])) &&
           isleft(p, QuadraticSegment_2D(tri6[2], tri6[3], tri6[5])) &&
           isleft(p, QuadraticSegment_2D(tri6[3], tri6[1], tri6[6]))
end

# # Slower method than above.
# function in(p::Point_2D, tri6::Triangle6_2D)
#     return in(p, tri6, 30)
# end
#
# function in(p::Point_2D, tri6::Triangle6_2D, N::Int64)
#     # Determine if the point is in the triangle using the Newton-Raphson method
#     # N is the max number of iterations of the method.
#     p_rs = real_to_parametric(p, tri6, N)
#     ϵ = parametric_coordinate_ϵ
#     # Check that the r coordinate and s coordinate are in [-ϵ,  1 + ϵ] and
#     # r + s ≤ 1 + ϵ
#     # These are the conditions for a valid point in the triangle ± some ϵ
#     if (-ϵ ≤ p_rs[1] ≤ 1 + ϵ) &&
#        (-ϵ ≤ p_rs[2] ≤ 1 + ϵ) &&
#        (p_rs[1] + p_rs[2] ≤ 1 + ϵ)
#         return true
#     else
#         return false
#     end
# end

function Base.intersect(l::LineSegment_2D{T}, tri6::Triangle6_2D{T}) where {T}
    # Create the 3 line segments that make up the triangle and intersect each one
    p₁ = Point_2D{T}(0,0)
    p₂ = Point_2D{T}(0,0)
    p₃ = Point_2D{T}(0,0)
    p₄ = Point_2D{T}(0,0)
    p₅ = Point_2D{T}(0,0)
    p₆ = Point_2D{T}(0,0)
    npoints = 0x0000
    for i ∈ 1:3
        hits, points = l ∩ QuadraticSegment_2D(tri6[(i - 1) % 3 + 1], 
                                               tri6[      i % 3 + 1],
                                               tri6[          i + 3])
        for point in view(points, 1:Int64(hits))
            npoints += 0x0001
            if npoints === 0x0001
                p₁ = point
            elseif npoints === 0x0002
                p₂ = point
            elseif npoints === 0x0003
                p₃ = point
            elseif npoints === 0x0004
                p₄ = point
            elseif npoints === 0x0005
                p₅ = point
            else
                p₆ = point
            end
        end
    end
    return npoints, SVector(p₁, p₂, p₃, p₄, p₅, p₆) 
end

function real_to_parametric(p::Point_2D, tri6::Triangle6_2D)
    return real_to_parametric(p, tri6, 30)
end

function real_to_parametric(p::Point_2D{T}, tri6::Triangle6_2D{T}, max_iters::Int64) where {T}
    # Convert from real coordinates to the triangle's local parametric coordinates using the
    # the Newton-Raphson method. 
    # If a conversion doesn't exist, the minimizer is returned.
    # Initial guess at triangle centroid
    rs = SVector{2,T}(1//3, 1//3) + inv(𝗝(tri6, 1//3, 1//3))*(p - tri6(1//3, 1//3))
    for i ∈ 1:max_iters
        Δrs = inv(𝗝(tri6, rs[1], rs[2]))*(p - tri6(rs[1], rs[2])) 
        if Δrs ⋅ Δrs < T((1e-6)^2)
            break
        end
        rs += Δrs
    end
    return Point_2D{T}(rs[1], rs[2])
end

function triangulate(tri6::Triangle6{N,T}, D::Int64) where {N,T}
    # D is the number of divisions of each edge
    triangles = Vector{Triangle{N,T}}(undef, (D+1)*(D+1))
    if D === 0
        triangles[1] = Triangle(tri6[1], tri6[2], tri6[3])
    else
        i = 1
        for s = 1:D, r = 0:D-s
            triangles[i]   = Triangle(tri6(    r/(D+1),     s/(D+1)),
                                      tri6((r+1)/(D+1),     s/(D+1)),
                                      tri6(    r/(D+1), (s+1)/(D+1)))
            triangles[i+1] = Triangle(tri6(    r/(D+1),     s/(D+1)),
                                      tri6((r+1)/(D+1), (s-1)/(D+1)),
                                      tri6((r+1)/(D+1),     s/(D+1)))
            i += 2
        end
        j = (D+1)*D + 1
        for s = 0:0, r = 0:D-s
            triangles[j] = Triangle(tri6(    r/(D+1),     s/(D+1)),
                                    tri6((r+1)/(D+1),     s/(D+1)),
                                    tri6(    r/(D+1), (s+1)/(D+1)))
            j += 1
        end
    end
    return triangles
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, tri6::Triangle6)
        q₁ = QuadraticSegment(tri6[1], tri6[2], tri6[4])
        q₂ = QuadraticSegment(tri6[2], tri6[3], tri6[5])
        q₃ = QuadraticSegment(tri6[3], tri6[1], tri6[6])
        qsegs = [q₁, q₂, q₃]
        return convert_arguments(LS, qsegs)
    end

    function convert_arguments(LS::Type{<:LineSegments}, T::Vector{<:Triangle6})
        point_sets = [convert_arguments(LS, tri6) for tri6 ∈ T]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(P::Type{<:Mesh}, tri6::Triangle6_2D)
        triangles = triangulate(tri6, 13)
        return convert_arguments(P, triangles)
    end

    function convert_arguments(M::Type{<:Mesh}, T::Vector{Triangle6_2D})
        triangles = reduce(vcat, triangulate.(T, 13))
        return convert_arguments(M, triangles)
    end
end
