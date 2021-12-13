# A quadratic quadrilateral in 2D space.
struct Quadrilateral8_2D{F <: AbstractFloat} <: Face_2D{F}
    # Fhe points are assumed to be ordered  in counter clockwise order as follows
    # p₁ = vertex A
    # p₂ = vertex B
    # p₃ = vertex C
    # p₄ = vertex D
    # p₅ = point on the quadratic segment from A to B
    # p₆ = point on the quadratic segment from B to C
    # p₇ = point on the quadratic segment from C to D
    # p₈ = point on the quadratic segment from D to A
    points::SVector{8, Point_2D{F}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral8_2D(p₁::Point_2D, p₂::Point_2D, p₃::Point_2D, p₄::Point_2D,
                  p₅::Point_2D, p₆::Point_2D, p₇::Point_2D, p₈::Point_2D
                 ) = Quadrilateral8_2D(SVector(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

# Methods (All type-stable)
# -------------------------------------------------------------------------------------------------
function (quad8::Quadrilateral8_2D{F})(r::R, s::S) where {F <: AbstractFloat,
                                                          R <: Real,
                                                          S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    ξ = 2F(r) - 1; η = 2F(s) - 1
    return (1 - ξ)*(1 - η)*(-ξ - η - 1)/4*quad8.points[1] +
           (1 + ξ)*(1 - η)*( ξ - η - 1)/4*quad8.points[2] +
           (1 + ξ)*(1 + η)*( ξ + η - 1)/4*quad8.points[3] +
           (1 - ξ)*(1 + η)*(-ξ + η - 1)/4*quad8.points[4] +
                      (1 - ξ^2)*(1 - η)/2*quad8.points[5] +
                      (1 - η^2)*(1 + ξ)/2*quad8.points[6] +
                      (1 - ξ^2)*(1 + η)/2*quad8.points[7] +
                      (1 - η^2)*(1 - ξ)/2*quad8.points[8]
end

# Interpolation with a point, instead of (r,s)
function (quad8::Quadrilateral8_2D{F})(p::Point_2D{F}) where {F <: AbstractFloat}
    r = p[1]; s = p[2]
    ξ = 2r - 1; η = 2s - 1
    return (1 - ξ)*(1 - η)*(-ξ - η - 1)/4*quad8.points[1] +
           (1 + ξ)*(1 - η)*( ξ - η - 1)/4*quad8.points[2] +
           (1 + ξ)*(1 + η)*( ξ + η - 1)/4*quad8.points[3] +
           (1 - ξ)*(1 + η)*(-ξ + η - 1)/4*quad8.points[4] +
                      (1 - ξ^2)*(1 - η)/2*quad8.points[5] +
                      (1 - η^2)*(1 + ξ)/2*quad8.points[6] +
                      (1 - ξ^2)*(1 + η)/2*quad8.points[7] +
                      (1 - η^2)*(1 - ξ)/2*quad8.points[8]
end

function derivative(quad8::Quadrilateral8_2D{F}, r::R, s::S) where {F <: AbstractFloat,
                                                                    R <: Real,
                                                                    S <: Real}
    # Chain rule
    # ∂Q   ∂Q ∂ξ     ∂Q      ∂Q   ∂Q ∂η     ∂Q
    # -- = -- -- = 2 -- ,    -- = -- -- = 2 --
    # ∂r   ∂ξ ∂r     ∂ξ      ∂s   ∂η ∂s     ∂η
    ξ = 2F(r) - 1; η = 2F(s) - 1
    ∂Q_∂ξ = (1 - η)*(2ξ + η)/4*quad8.points[1] +
            (1 - η)*(2ξ - η)/4*quad8.points[2] +
            (1 + η)*(2ξ + η)/4*quad8.points[3] +
            (1 + η)*(2ξ - η)/4*quad8.points[4] +
                    -ξ*(1 - η)*quad8.points[5] +
                   (1 - η^2)/2*quad8.points[6] +
                    -ξ*(1 + η)*quad8.points[7] +
                  -(1 - η^2)/2*quad8.points[8]

    ∂Q_∂η = (1 - ξ)*( ξ + 2η)/4*quad8.points[1] +
            (1 + ξ)*(-ξ + 2η)/4*quad8.points[2] +
            (1 + ξ)*( ξ + 2η)/4*quad8.points[3] +
            (1 - ξ)*(-ξ + 2η)/4*quad8.points[4] +
                   -(1 - ξ^2)/2*quad8.points[5] +
                     -η*(1 + ξ)*quad8.points[6] +
                    (1 - ξ^2)/2*quad8.points[7] +
                     -η*(1 - ξ)*quad8.points[8]

    return 2*∂Q_∂ξ, 2*∂Q_∂η
end

function jacobian(quad8::Quadrilateral8_2D, r::R, s::S) where {R <: Real,
                                                               S <: Real}
    # Return the 2 x 2 Jacobian matrix
    ∂Q_∂r, ∂Q_∂s = derivative(quad8, r, s)
    return SMatrix{2, 2}(∂Q_∂r.x, ∂Q_∂r.y,
                         ∂Q_∂s.x, ∂Q_∂s.y)
end

function area(quad8::Quadrilateral8_2D{F}) where {F <: AbstractFloat}
    return area(quad8, Val(3))
end

function area(quad8::Quadrilateral8_2D{F}, ::Val{N}) where {N, F <: AbstractFloat}
    # Numerical integration required. Gauss-Legendre quadrature over a quadrilateral is used.
    # Let Q(r,s) be the interpolation function for quad8,
    #                             1  1
    # A = ∬ ||∂Q/∂r × ∂Q/∂s||dA = ∫  ∫ ||∂Q/∂r × ∂Q/∂s|| ds dr
    #      D                      0  0
    #
    #       N   N
    #   =   ∑   ∑  wᵢwⱼ||∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)||
    #      i=1 j=1
    # N is the square root of the number of points used in the quadrature.
    # See tuning/Quadrilateral8_2D_area.jl for more info on how N = 3 was chosen.
    w, r = gauss_legendre_quadrature(F, Val(N))
    a = F(0)
    for i = 1:N, j = 1:N
        ∂Q_∂r, ∂Q_∂s = derivative(quad8, r[i], r[j])
        a += w[i]*w[j]*abs(∂Q_∂r × ∂Q_∂s)
    end
    return a
end

function triangulate(quad8::Quadrilateral8_2D{F}, N::Int64) where {F <: AbstractFloat}
    # N is the number of divisions of each edge
    triangles = Vector{Triangle_2D{F}}(undef, 2*(N+1)*(N+1))
    if N === 0
        triangles[1] = Triangle_2D(quad8.points[1], quad8.points[2], quad8.points[3])
        triangles[2] = Triangle_2D(quad8.points[3], quad8.points[4], quad8.points[1])
    else
        for j = 0:N, i = 0:N
            triangles[2*(N+1)*j + 2i + 1] = Triangle_2D(quad8(    i/(N+1),     j/(N+1)),
                                                        quad8((i+1)/(N+1),     j/(N+1)),
                                                        quad8(    i/(N+1), (j+1)/(N+1)))
            triangles[2*(N+1)*j + 2i + 2] = Triangle_2D(quad8(    i/(N+1), (j+1)/(N+1)),
                                                        quad8((i+1)/(N+1),     j/(N+1)),
                                                        quad8((i+1)/(N+1), (j+1)/(N+1)))
        end
    end
    return triangles
end

function real_to_parametric(p::Point_2D{F}, quad8::Quadrilateral8_2D{F}) where {F <: AbstractFloat}
    return real_to_parametric(p, quad8, 30)
end

function real_to_parametric(p::Point_2D{F},
                            quad8::Quadrilateral8_2D{F}, N::Int64) where {F <: AbstractFloat}
    # Convert from real coordinates to the triangle's local parametric coordinates using the
    # the Newton-Raphson method. N is the max number of iterations
    # If a conversion doesn't exist, the minimizer is returned.
    r = F(1//2) # Initial guess at centroid
    s = F(1//2)
    err₁ = p - quad8(r, s)
    for i = 1:N
        # Inversion is faster for 2 by 2 than \
        Δr, Δs = inv(jacobian(quad8, r, s)) * err₁
        r = r + Δr
        s = s + Δs
        err₂ = p - quad8(r, s)
        if norm(err₂ - err₁) < 1.0e-6
            break
        end
        err₁ = err₂
    end
    return Point_2D(r, s)
end

function in(p::Point_2D{F}, quad8::Quadrilateral8_2D{F}) where {F <: AbstractFloat}
    # If the point is to the left of every edge
    #  4<-----3
    #  |      ^
    #  | p    |
    #  |      |
    #  |      |
    #  v----->2
    #  1
    return is_left(p, QuadraticSegment_2D(quad8.points[1], quad8.points[2], quad8.points[5])) &&
           is_left(p, QuadraticSegment_2D(quad8.points[2], quad8.points[3], quad8.points[6])) &&
           is_left(p, QuadraticSegment_2D(quad8.points[3], quad8.points[4], quad8.points[7])) &&
           is_left(p, QuadraticSegment_2D(quad8.points[4], quad8.points[1], quad8.points[8]))
end

# function in(p::Point_2D{F}, quad8::Quadrilateral8_2D{F}, N::Int64) where {F <: AbstractFloat}
#     # Determine if the point is in the triangle using the Newton-Raphson method
#     # N is the max number of iterations of the method.
#     p_rs = real_to_parametric(p, quad8, N)
#     ϵ = parametric_coordinate_ϵ
#     if (-ϵ ≤ p_rs[1] ≤ 1 + ϵ) &&
#        (-ϵ ≤ p_rs[2] ≤ 1 + ϵ)
#         return true
#     else
#         return false
#     end
# end

function intersect(l::LineSegment_2D{F}, quad8::Quadrilateral8_2D{F}) where {F <: AbstractFloat}
    # Create the 3 quadratic segments that make up the triangle and intersect each one
    edges = SVector(QuadraticSegment_2D(quad8.points[1], quad8.points[2], quad8.points[5]),
                    QuadraticSegment_2D(quad8.points[2], quad8.points[3], quad8.points[6]),
                    QuadraticSegment_2D(quad8.points[3], quad8.points[4], quad8.points[7]),
                    QuadraticSegment_2D(quad8.points[4], quad8.points[1], quad8.points[8]))
    ipoints = MVector(Point_2D(F, 0),
                      Point_2D(F, 0),
                      Point_2D(F, 0),
                      Point_2D(F, 0),
                      Point_2D(F, 0),
                      Point_2D(F, 0)
                     )
    n_ipoints = 0x00000000
    # We need to account for 6 points returned
    for k = 1:4                              
        npoints, points = l ∩ edges[k]
        for i = 1:npoints
            n_ipoints += 0x00000001 
            ipoints[n_ipoints] = points[i]
        end
    end
    return n_ipoints, SVector(ipoints)
end
intersect(quad8::Quadrilateral8_2D, l::LineSegment_2D) = intersect(l, quad8)

function Base.show(io::IO, quad8::Quadrilateral8_2D{F}) where {F <: AbstractFloat}
    println(io, "Quadrilateral8_2D{$F}(")
    for i = 1:8
        p = quad8.points[i]
        println(io, "  $p,")
    end
    println(io, " )")
end

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, quad8::Quadrilateral8_2D)
        q₁ = QuadraticSegment_2D(quad8.points[1], quad8.points[2], quad8.points[5])
        q₂ = QuadraticSegment_2D(quad8.points[2], quad8.points[3], quad8.points[6])
        q₃ = QuadraticSegment_2D(quad8.points[3], quad8.points[4], quad8.points[7])
        q₄ = QuadraticSegment_2D(quad8.points[4], quad8.points[1], quad8.points[8])
        qsegs = [q₁, q₂, q₃, q₄]
        return convert_arguments(LS, qsegs)
    end
    
    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{Quadrilateral8_2D})
        point_sets = [convert_arguments(LS, quad8) for quad8 in Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
    end
    
    function convert_arguments(M::Type{<:Mesh}, quad8::Quadrilateral8_2D)
        triangles = triangulate(quad8, 13)
        return convert_arguments(M, triangles)
    end
    
    function convert_arguments(M::Type{<:Mesh}, Q::Vector{Quadrilateral8_2D})
        triangles = reduce(vcat, triangulate.(Q, 13))
        return convert_arguments(M, triangles)
    end
end
