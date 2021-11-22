# @code_warntype checked 2021/11/20

# A quadratic triangle, defined in 2D.
struct Triangle6_2D{F <: AbstractFloat} <: Face_2D{F}
    # The points are assumed to be ordered as follows
    # p₁ = vertex A
    # p₂ = vertex B
    # p₃ = vertex C
    # p₄ = point on the quadratic segment from A to B
    # p₅ = point on the quadratic segment from B to C
    # p₆ = point on the quadratic segment from C to A
    points::NTuple{6, Point_2D{F}}

end

# Constructors
# -------------------------------------------------------------------------------------------------
# @code_warntype checked 2021/11/20
Triangle6_2D(p₁::Point_2D,
             p₂::Point_2D,
             p₃::Point_2D,
             p₄::Point_2D,
             p₅::Point_2D,
             p₆::Point_2D
            ) = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))


# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
# @code_warntype checked 2021/11/20
function (tri6::Triangle6_2D{F})(r::R, s::S) where {F <: AbstractFloat,
                                                    R <: Real,
                                                    S <: Real}
    # See Fhe Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = F(r)
    sₜ = F(s)
    return (1 - rₜ - sₜ)*(2(1 - rₜ - sₜ) - 1)*tri6.points[1] +
                                   rₜ*(2rₜ-1)*tri6.points[2] +
                                   sₜ*(2sₜ-1)*tri6.points[3] +
                            4rₜ*(1 - rₜ - sₜ)*tri6.points[4] +
                                     (4rₜ*sₜ)*tri6.points[5] +
                            4sₜ*(1 - rₜ - sₜ)*tri6.points[6]
end

# Interpolation using a point instead of (r,s)
# @code_warntype checked 2021/11/20
function (tri6::Triangle6_2D{F})(p::Point_2D{F}) where {F <: AbstractFloat}
    r = p[1]
    s = p[2]
    return (1 - r - s)*(2(1 - r - s) - 1)*tri6.points[1] +
                                 r*(2r-1)*tri6.points[2] +
                                 s*(2s-1)*tri6.points[3] +
                           4r*(1 - r - s)*tri6.points[4] +
                                   (4r*s)*tri6.points[5] +
                           4s*(1 - r - s)*tri6.points[6]
end

# @code_warntype checked 2021/11/20
function derivative(tri6::Triangle6_2D{F}, r::R, s::S) where {F <: AbstractFloat,
                                                              R <: Real,
                                                              S <: Real}
    # Let F(r,s) be the interpolation function for tri6
    # Returns ∂F/∂r, ∂F/∂s
    rₜ = F(r)
    sₜ = F(s)
    ∂F_∂r = (4rₜ + 4sₜ - 3)*tri6.points[1] +
                  (4rₜ - 1)*tri6.points[2] +
            4(1 - 2rₜ - sₜ)*tri6.points[4] +
                      (4sₜ)*tri6.points[5] +
                     (-4sₜ)*tri6.points[6]

    ∂F_∂s = (4rₜ + 4sₜ - 3)*tri6.points[1] +
                  (4sₜ - 1)*tri6.points[3] +
                     (-4rₜ)*tri6.points[4] +
                      (4rₜ)*tri6.points[5] +
            4(1 - rₜ - 2sₜ)*tri6.points[6]
    return ∂F_∂r, ∂F_∂s
end

# @code_warntype checked 2021/11/20
function jacobian(tri6::Triangle6_2D, r::R, s::S) where {R <: Real,
                                                         S <: Real}
    # Return the 2 x 2 Jacobian matrix
    ∂F_∂r, ∂F_∂s = derivative(tri6, r, s)
    return hcat(∂F_∂r.x, ∂F_∂s.x)
end

# @code_warntype checked 2021/11/20
function area(tri6::Triangle6_2D{F}; N::Int64=12) where {F <: AbstractFloat}
    # Numerical integration required. Gauss-Legendre quadrature over a triangle is used.
    # Let F(r,s) be the interpolation function for tri6,
    #                             1 1-r                          N
    # A = ∬ ||∂F/∂r × ∂F/∂s||dA = ∫  ∫ ||∂F/∂r × ∂F/∂s|| ds dr = ∑ wᵢ||∂F/∂r(rᵢ,sᵢ) × ∂F/∂s(rᵢ,sᵢ)||
    #      D                      0  0                          i=1
    #
    # N is the number of points used in the quadrature.
    # See tuning/Triangle6_2D_area.jl for more info on how N = 12 was chosen.
    w, r, s = gauss_legendre_quadrature(tri6, N)
    a = F(0)
    for i in 1:N
        ∂F_∂r, ∂F_∂s = derivative(tri6, r[i], s[i])
        a += w[i] * abs(∂F_∂r × ∂F_∂s)
    end
    return a
end

# @code_warntype checked 2021/11/20
function triangulate(tri6::Triangle6_2D{F}, N::Int64) where {F <: AbstractFloat}
    # N is the number of divisions of each edge
    triangles = Vector{Triangle_2D{F}}(undef, (N+1)*(N+1))
    if N === 0
        triangles[1] = Triangle_2D(tri6.points[1], tri6.points[2], tri6.points[3])
    else
        i = 1
        for S = 1:N, R = 0:N-S
            triangles[i]   = Triangle_2D(tri6(    R/(N+1),     S/(N+1)),
                                         tri6((R+1)/(N+1),     S/(N+1)),
                                         tri6(    R/(N+1), (S+1)/(N+1)))
            triangles[i+1] = Triangle_2D(tri6(    R/(N+1),     S/(N+1)),
                                         tri6((R+1)/(N+1), (S-1)/(N+1)),
                                         tri6((R+1)/(N+1),     S/(N+1)))
            i += 2
        end
        j = (N+1)*N + 1
        for S = 0:0, R = 0:N-S
            triangles[j] = Triangle_2D(tri6(    R/(N+1),     S/(N+1)),
                                       tri6((R+1)/(N+1),     S/(N+1)),
                                       tri6(    R/(N+1), (S+1)/(N+1)))
            j += 1
        end
    end
    return triangles
end

# @code_warntype checked 2021/11/20
function real_to_parametric(p::Point_2D{F}, tri6::Triangle6_2D{F}; N::Int64=30) where {F <: AbstractFloat}
    # Convert from real coordinates to the triangle's local parametric coordinates using the
    # the Newton-Raphson method. N is the max number of iterations
    # If a conversion doesn't exist, the minimizer is returned.
    r = F(1//3) # Initial guess at triangle centroid
    s = F(1//3)
    err₁ = p - tri6(r, s)
    for i = 1:N
        # Inversion is faster for 2 by 2 than \
        Δr, Δs = inv(jacobian(tri6, r, s)) * err₁.x
        r = r + Δr
        s = s + Δs
        err₂ = p - tri6(r, s)
        if norm(err₂ - err₁) < 1e-6
            break
        end
        err₁ = err₂
    end
    return Point_2D(r, s)
end

# @code_warntype checked 2021/11/20
function in(p::Point_2D, tri6::Triangle6_2D; N::Int64=30)
    # Determine if the point is in the triangle using the Newton-Raphson method
    # N is the max number of iterations of the method.
    p_rs = real_to_parametric(p, tri6; N=N)
    ϵ = parametric_coordinate_ϵ 
    # Check that the r coordinate and s coordinate are in [-ϵ,  1 + ϵ] and
    # r + s ≤ 1 + ϵ
    # These are the conditions for a valid point in the triangle ± some ϵ
    if (-ϵ ≤ p_rs[1] ≤ 1 + ϵ) &&
       (-ϵ ≤ p_rs[2] ≤ 1 + ϵ) &&
       (p_rs[1] + p_rs[2] ≤ 1 + ϵ)
        return true
    else
        return false
    end
end

# @code_warntype checked 2021/11/20
function intersect(l::LineSegment_2D{F}, tri6::Triangle6_2D{F}) where {F <: AbstractFloat}
    # Create the 3 quadratic segments that make up the triangle and intersect each one
    edges = (QuadraticSegment_2D(tri6.points[1], tri6.points[2], tri6.points[4]),
             QuadraticSegment_2D(tri6.points[2], tri6.points[3], tri6.points[5]),
             QuadraticSegment_2D(tri6.points[3], tri6.points[1], tri6.points[6]))
    ipoints = MVector(Point_2D(F, 0),
                      Point_2D(F, 0),
                      Point_2D(F, 0),
                      Point_2D(F, 0),
                      Point_2D(F, 0),
                      Point_2D(F, 0)
                     )
    n_ipoints = 0x00
    # We need to account for 6 points returned
    for k = 1:3
        npoints, points = l ∩ edges[k]
        for i = 1:npoints
            if n_ipoints === 0x00
                ipoints[1] = points[1]
                n_ipoints = 0x01
            else
                # make sure we don't have duplicate points
                duplicate = false
                for j = 1:n_ipoints
                    if points[i] ≈ ipoints[j]
                        duplicate = true
                        break
                    end
                end
                if !duplicate
                    n_ipoints += 0x01
                    ipoints[n_ipoints] = points[i]
                end
            end
        end
    end
    return n_ipoints, Tuple(ipoints)
end
intersect(tri6::Triangle6_2D, l::LineSegment_2D) = intersect(l, tri6)

function Base.show(io::IO, tri6::Triangle6_2D{F}) where {F <: AbstractFloat}
    println(io, "Triangle6_2D{$F}(")
    for i = 1:6
        p = tri6.points[i]
        println(io, "  $p,")
    end
    println(io, " )")
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(LS::Type{<:LineSegments}, tri6::Triangle6_2D)
    q₁ = QuadraticSegment_2D(tri6.points[1], tri6.points[2], tri6.points[4])
    q₂ = QuadraticSegment_2D(tri6.points[2], tri6.points[3], tri6.points[5])
    q₃ = QuadraticSegment_2D(tri6.points[3], tri6.points[1], tri6.points[6])
    qsegs = [q₁, q₂, q₃]
    return convert_arguments(P, qsegs)
end

function convert_arguments(LS::Type{<:LineSegments}, T::Vector{Triangle6_2D})
    point_sets = [convert_arguments(LS, tri6) for tri6 in T]
    return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{<:Mesh}, tri6::Triangle6_2D)
    triangles = triangulate(tri6, 13)
    return convert_arguments(P, triangles)
end

function convert_arguments(M::Type{<:Mesh}, T::Vector{Triangle6_2D})
    triangles = reduce(vcat, triangulate.(T, 13))
    return convert_arguments(M, triangles)
end
