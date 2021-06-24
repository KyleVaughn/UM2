# A quadratic quadrilateral in 2D space.

struct Quadrilateral8_2D{T <: AbstractFloat}
    # The points are assumed to be ordered  in counter clockwise order as follows
    # p₁ = vertex A
    # p₂ = vertex B
    # p₃ = vertex C
    # p₄ = vertex D
    # p₅ = point on the quadratic segment from A to B
    # p₆ = point on the quadratic segment from B to C
    # p₇ = point on the quadratic segment from C to D
    # p₈ = point on the quadratic segment from D to A
    points::NTuple{8, Point_2D{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral8_2D(p₁::Point_2D{T}, p₂::Point_2D{T}, p₃::Point_2D{T}, p₄::Point_2D{T},
                  p₅::Point_2D{T}, p₆::Point_2D{T}, p₇::Point_2D{T}, p₈::Point_2D{T}
        ) where {T <: AbstractFloat} = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

# Methods
# -------------------------------------------------------------------------------------------------
function (quad8::Quadrilateral8_2D{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    ξ = 2T(r) - 1; η = 2T(s) - 1
    return (1 - ξ)*(1 - η)*(-ξ - η - 1)/4*quad8.points[1] +
           (1 + ξ)*(1 - η)*( ξ - η - 1)/4*quad8.points[2] +
           (1 + ξ)*(1 + η)*( ξ + η - 1)/4*quad8.points[3] +
           (1 - ξ)*(1 + η)*(-ξ + η - 1)/4*quad8.points[4] +
                      (1 - ξ^2)*(1 - η)/2*quad8.points[5] +
                      (1 - η^2)*(1 + ξ)/2*quad8.points[6] +
                      (1 - ξ^2)*(1 + η)/2*quad8.points[7] +
                      (1 - η^2)*(1 - ξ)/2*quad8.points[8]
end

function (quad8::Quadrilateral8_2D{T})(p::Point_2D{T}) where {T <: AbstractFloat}
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

function derivatives(quad8::Quadrilateral8_2D{T}, r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    # Chain rule
    # ∂Q   ∂Q ∂ξ  ∂Q   ∂Q ∂η 
    # -- = -- --, -- = -- --
    # ∂r   ∂ξ ∂r  ∂s   ∂η ∂s
    ξ = 2T(r) - 1; η = 2T(s) - 1
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

function jacobian(quad8::Quadrilateral8_2D{T}, r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    # Return the 2 x 2 Jacobian matrix
    ∂Q_∂r, ∂Q_∂s = derivatives(quad8, r, s)
    return hcat(∂Q_∂r.x, ∂Q_∂s.x)
end

function area(quad8::Quadrilateral8_2D{T}; N::Int64=15) where {T <: AbstractFloat}
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
    # See tuning/Quadrilateral8_2D_area.jl for more info on how N = 15 was chosen.    
    w, r = gauss_legendre_quadrature(T, N)
    a = T(0)
    for i = 1:N, j = 1:N
        ∂Q_∂r, ∂Q_∂s = derivatives(quad8, r[i], r[j])
        a += w[i]*w[j]*norm(∂Q_∂r × ∂Q_∂s)
    end
    return a
end

function triangulate(quad8::Quadrilateral8_2D{T}, N::Int64) where {T <: AbstractFloat}
    # N is the number of divisions of each edge
    triangles = Vector{Triangle_2D{T}}(undef, 2*(N+1)*(N+1))
    if N == 0
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

# in, real_to_parametric

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, quad8::Quadrilateral8_2D{T}) where {T <: AbstractFloat}
    q₁ = QuadraticSegment_2D(quad8.points[1], quad8.points[2], quad8.points[5])
    q₂ = QuadraticSegment_2D(quad8.points[2], quad8.points[3], quad8.points[6])
    q₃ = QuadraticSegment_2D(quad8.points[3], quad8.points[4], quad8.points[7])
    q₄ = QuadraticSegment_2D(quad8.points[4], quad8.points[1], quad8.points[8])
    qsegs = [q₁, q₂, q₃, q₄]
    return convert_arguments(P, qsegs)
end

function convert_arguments(P::Type{<:LineSegments}, 
        QA::AbstractArray{<:Quadrilateral8_2D{T}}) where {T <: AbstractFloat}
    point_sets = [convert_arguments(P, quad8) for quad8 in QA]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{Mesh{Tuple{Quadrilateral8_2D{T}}}}, 
        quad8::Quadrilateral8_2D{T}) where {T <: AbstractFloat}
    triangles = triangulate(quad8, 13)
    return convert_arguments(P, triangles)
end

#function convert_arguments(MT::Type{Mesh{Tuple{Quadrilateral8_2D{T}}}},
#        AQ::Vector{Triangle{T}}) where {T <: AbstractFloat}
#    points = reduce(vcat, [[quad8.points[i].coord for i = 1:4] for quad8 in AQ])
#    faces = zeros(Int64, length(AT), 3)
#    k = 1
#    for i in 1:length(AT), j = 1:3
#        faces[i, j] = k
#        k += 1
#    end
#    return convert_arguments(MT, points, faces)
#end