# A quadratic triangle, defined in 2D.
struct Triangle6_2D{T <: AbstractFloat} <: Face_2D{T}
    # The points are assumed to be ordered as follows
    # p₁ = vertex A
    # p₂ = vertex B
    # p₃ = vertex C
    # p₄ = point on the quadratic segment from A to B
    # p₅ = point on the quadratic segment from B to C
    # p₆ = point on the quadratic segment from C to A
    points::NTuple{6, Point_2D{T}}

end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle6_2D(p₁::Point_2D{T},
             p₂::Point_2D{T},
             p₃::Point_2D{T},
             p₄::Point_2D{T},
             p₅::Point_2D{T},
             p₆::Point_2D{T}
            ) where {T <: AbstractFloat} = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))


# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
function (tri6::Triangle6_2D{T})(r::R, s::S) where {T <: AbstractFloat,
                                                    R <: Real,
                                                    S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = T(r)
    sₜ = T(s)
    return (1 - rₜ - sₜ)*(2(1 - rₜ - sₜ) - 1)*tri6.points[1] +
                                   rₜ*(2rₜ-1)*tri6.points[2] +
                                   sₜ*(2sₜ-1)*tri6.points[3] +
                            4rₜ*(1 - rₜ - sₜ)*tri6.points[4] +
                                     (4rₜ*sₜ)*tri6.points[5] +
                            4sₜ*(1 - rₜ - sₜ)*tri6.points[6]
end

# Interpolation using a point instead of (r,s)
function (tri6::Triangle6_2D{T})(p::Point_2D{T}) where {T <: AbstractFloat}
    r = p[1]
    s = p[2]
    return (1 - r - s)*(2(1 - r - s) - 1)*tri6.points[1] +
                                 r*(2r-1)*tri6.points[2] +
                                 s*(2s-1)*tri6.points[3] +
                           4r*(1 - r - s)*tri6.points[4] +
                                   (4r*s)*tri6.points[5] +
                           4s*(1 - r - s)*tri6.points[6]
end

function derivative(tri6::Triangle6_2D{T}, r::R, s::S) where {T <: AbstractFloat,
                                                               R <: Real,
                                                               S <: Real}
    # Let T(r,s) be the interpolation function for tri6
    # Returns ∂T/∂r, ∂T/∂s
    rₜ = T(r)
    sₜ = T(s)
    ∂T_∂r = (4rₜ + 4sₜ - 3)*tri6.points[1] +
                  (4rₜ - 1)*tri6.points[2] +
            4(1 - 2rₜ - sₜ)*tri6.points[4] +
                      (4sₜ)*tri6.points[5] +
                     (-4sₜ)*tri6.points[6]

    ∂T_∂s = (4rₜ + 4sₜ - 3)*tri6.points[1] +
                  (4sₜ - 1)*tri6.points[3] +
                     (-4rₜ)*tri6.points[4] +
                      (4rₜ)*tri6.points[5] +
            4(1 - rₜ - 2sₜ)*tri6.points[6]
    return ∂T_∂r, ∂T_∂s
end

function jacobian(tri6::Triangle6_2D{T}, r::R, s::S) where {T <: AbstractFloat,
                                                            R <: Real,
                                                            S <: Real}
    # Return the 2 x 2 Jacobian matrix
    ∂T_∂r, ∂T_∂s = derivative(tri6, r, s)
    return hcat(∂T_∂r.x, ∂T_∂s.x)
end

function area(tri6::Triangle6_2D{T}; N::Int64=12) where {T <: AbstractFloat}
    # Numerical integration required. Gauss-Legendre quadrature over a triangle is used.
    # Let T(r,s) be the interpolation function for tri6,
    #                             1 1-r                          N
    # A = ∬ ||∂T/∂r × ∂T/∂s||dA = ∫  ∫ ||∂T/∂r × ∂T/∂s|| ds dr = ∑ wᵢ||∂T/∂r(rᵢ,sᵢ) × ∂T/∂s(rᵢ,sᵢ)||
    #      D                      0  0                          i=1
    #
    # N is the number of points used in the quadrature.
    # See tuning/Triangle6_2D_area.jl for more info on how N = 12 was chosen.
    w, r, s = gauss_legendre_quadrature(tri6, N)
    a = T(0)
    for i in 1:N
        ∂T_∂r, ∂T_∂s = derivative(tri6, r[i], s[i])
        a += w[i] * abs(∂T_∂r × ∂T_∂s)
    end
    return a::T
end

function triangulate(tri6::Triangle6_2D{T}, N::Int64) where {T <: AbstractFloat}
    # N is the number of divisions of each edge
    triangles = Vector{Triangle_2D{T}}(undef, (N+1)*(N+1))
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

function real_to_parametric(p::Point_2D{T}, tri6::Triangle6_2D{T}; N::Int64=30) where {T <: AbstractFloat}
    # Convert from real coordinates to the triangle's local parametric coordinates using the
    # the Newton-Raphson method. N is the max number of iterations
    # If a conversion doesn't exist, the minimizer is returned.
    r = T(1//3) # Initial guess at triangle centroid
    s = T(1//3)
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

function in(p::Point_2D{T}, tri6::Triangle6_2D{T}; N::Int64=30) where {T <: AbstractFloat}
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

function intersect(l::LineSegment_2D{T}, tri6::Triangle6_2D{T}) where {T <: AbstractFloat}
    # Create the 3 quadratic segments that make up the triangle and intersect each one
    edges = (QuadraticSegment_2D(tri6.points[1], tri6.points[2], tri6.points[4]),
             QuadraticSegment_2D(tri6.points[2], tri6.points[3], tri6.points[5]),
             QuadraticSegment_2D(tri6.points[3], tri6.points[1], tri6.points[6]))
    intersections = l .∩ edges
    ipoints = MVector(Point_2D(T, 0),
                      Point_2D(T, 0),
                      Point_2D(T, 0),
                      Point_2D(T, 0),
                      Point_2D(T, 0),
                      Point_2D(T, 0)
                     )
    n_ipoints = 0x00
    # We need to account for 6 points returned
    for (npoints, points) in intersections
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

function Base.show(io::IO, tri6::Triangle6_2D{T}) where {T <: AbstractFloat}
    println(io, "Triangle6_2D{$T}(")
    for i = 1:6
        p = tri6.points[i]
        println(io, "  $p,")
    end
    println(io, " )")
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, tri6::Triangle6_2D{T}) where {T <: AbstractFloat}
    q₁ = QuadraticSegment_2D(tri6.points[1], tri6.points[2], tri6.points[4])
    q₂ = QuadraticSegment_2D(tri6.points[2], tri6.points[3], tri6.points[5])
    q₃ = QuadraticSegment_2D(tri6.points[3], tri6.points[1], tri6.points[6])
    qsegs = [q₁, q₂, q₃]
    return convert_arguments(P, qsegs)
end

function convert_arguments(P::Type{<:LineSegments},
        TA::AbstractArray{<:Triangle6_2D{T}}) where {T <: AbstractFloat}
    point_sets = [convert_arguments(P, tri6) for tri6 in TA]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{Mesh{Tuple{Triangle6_2D{T}}}},
        tri6::Triangle6_2D{T}) where {T <: AbstractFloat}
    triangles = triangulate(tri6, 13)
    return convert_arguments(P, triangles)
end

function convert_arguments(MT::Type{Mesh{Tuple{Vector{Triangle6_2D{T}}}}},
        AT::Vector{Triangle6_2D{T}}) where {T <: AbstractFloat}
    triangles = reduce(vcat, triangulate.(AT, 13))
    return convert_arguments(MT, triangles)
end
