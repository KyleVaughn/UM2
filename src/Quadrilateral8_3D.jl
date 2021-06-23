struct Quadrilateral8_3D{T <: AbstractFloat}
    points::NTuple{8, Point_3D{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral8_3D(p₁::Point_3D{T}, p₂::Point_3D{T}, p₃::Point_3D{T}, p₄::Point_3D{T},
                  p₅::Point_3D{T}, p₆::Point_3D{T}, p₇::Point_3D{T}, p₈::Point_3D{T}
        ) where {T <: AbstractFloat} = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

# Methods
# -------------------------------------------------------------------------------------------------
function (quad8::Quadrilateral8_3D{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
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

function derivatives(quad8::Quadrilateral8_3D{T}, r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    ξ = 2T(r) - 1; η = 2T(s) - 1
    d_dξ = (1 - η)*(2ξ + η)/4*quad8.points[1] + 
           (1 - η)*(2ξ - η)/4*quad8.points[2] +
           (1 + η)*(2ξ + η)/4*quad8.points[3] +     
           (1 + η)*(2ξ - η)/4*quad8.points[4] +     
                   -ξ*(1 - η)*quad8.points[5] +     
                  (1 - η^2)/2*quad8.points[6] +     
                   -ξ*(1 + η)*quad8.points[7] +     
                 -(1 - η^2)/2*quad8.points[8]     

    d_dη = (1 - ξ)*( ξ + 2η)/4*quad8.points[1] + 
           (1 + ξ)*(-ξ + 2η)/4*quad8.points[2] +
           (1 + ξ)*( ξ + 2η)/4*quad8.points[3] +     
           (1 - ξ)*(-ξ + 2η)/4*quad8.points[4] +     
                  -(1 - ξ^2)/2*quad8.points[5] +   
                    -η*(1 + ξ)*quad8.points[6] +     
                   (1 - ξ^2)/2*quad8.points[7] +     
                    -η*(1 - ξ)*quad8.points[8]     

    return 2*d_dξ, 2*d_dη 
end

function area(quad8::Quadrilateral8_3D{T}; N::Int64=15) where {T <: AbstractFloat}
    # Numerical integration required. Gauss-Legendre quadrature over a quadrilateral is used.
    # Let T(r,s) be the interpolation function for quad8,
    #                             1  1                         
    # A = ∬ ||∂T/∂r × ∂T/∂s||dA = ∫  ∫ ||∂T/∂r × ∂T/∂s|| ds dr 
    #      D                      0  0                         
    #
    #     1  1                         
    #   = ∫  ∫ ||∂T/∂ξ × ∂T/∂η|| |J| dξ dη, where |J| = Jacobian determinant = 4  
    #    -1 -1                         
    #
    #       N   N  
    #   = 4 ∑   ∑  wᵢwⱼ||∂T/∂ξ(ξᵢ,ηⱼ) × ∂T/∂η(ξᵢ,ηⱼ)||
    #      i=1 j=1
    # NOTE: for 2D, N = 4  appears to be sufficient. For 3D, N = 15  is preferred.
    # This is to ensure error in area less that about 1e-6. This was determined
    # experimentally, not mathematically, so more sophisticated analysis could be 
    # performed.
    W, R = gauss_legendre_quadrature(T, N)
    a = T(0)
    for i in 1:N, j = 1:N
        dξ, dη = derivatives(quad8, R[i], R[j])
        a += W[i]*W[j]*norm(dξ × dη)
    end
    return a
end

function triangulate(quad8::Quadrilateral8_3D{T}, N::Int64) where {T <: AbstractFloat}
    triangles = Vector{Triangle_3D{T}}(undef, 2*(N+1)*(N+1))
    if N == 0
        triangles[1] = Triangle_3D(quad8.points[1], quad8.points[2], quad8.points[3])
        triangles[2] = Triangle_3D(quad8.points[3], quad8.points[4], quad8.points[1])
    else
        for j = 0:N, i = 0:N
            triangles[2*(N+1)*j + 2i + 1] = Triangle_3D(quad8(    i/(N+1),     j/(N+1)), 
                                                        quad8((i+1)/(N+1),     j/(N+1)), 
                                                        quad8(    i/(N+1), (j+1)/(N+1)))
            triangles[2*(N+1)*j + 2i + 2] = Triangle_3D(quad8(    i/(N+1), (j+1)/(N+1)), 
                                                        quad8((i+1)/(N+1),     j/(N+1)), 
                                                        quad8((i+1)/(N+1), (j+1)/(N+1)))
        end
    end
    return triangles
end

function intersect(l::LineSegment_3D{T}, quad8::Quadrilateral8_3D{T}; N::Int64 = 13) where {T <: AbstractFloat}
    triangles = triangulate(quad8, N)
    npoints = 0 
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 0)
    intersections = l .∩ triangles
    bools = map(x->x[1], intersections)
    points = map(x->x[2], intersections)
    npoints = count(bools)
    if npoints == 0
        return false, 0, p₁, p₂
    elseif npoints == 1
        p₁ = points[argmax(bools)]
        return true, 1, p₁, p₂
    elseif npoints == 2
        indices = findall(bools)
        p₁ = points[indices[1]]
        p₂ = points[indices[2]]
        # Check uniqueness
        if p₁ ≈ p₂
            return true, 1, p₁, p₂
        else
            return true, 2, p₁, p₂
        end
    else
        # Account for 3 points and 4 points?
        # If intersection is on edge shared by two triangles on entrance and/or exit 3/4 intersections
        # can be detected
        return true, -1, p₁, p₂  
    end 
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, quad8::Quadrilateral8_3D{T}) where {T <: AbstractFloat}
    q₁ = QuadraticSegment_3D(quad8.points[1], quad8.points[2], quad8.points[5])
    q₂ = QuadraticSegment_3D(quad8.points[2], quad8.points[3], quad8.points[6])
    q₃ = QuadraticSegment_3D(quad8.points[3], quad8.points[4], quad8.points[7])
    q₄ = QuadraticSegment_3D(quad8.points[4], quad8.points[1], quad8.points[8])
    qsegs = [q₁, q₂, q₃, q₄]
    return convert_arguments(P, qsegs)
end

function convert_arguments(P::Type{<:LineSegments}, 
        QA::AbstractArray{<:Quadrilateral8_3D{T}}) where {T <: AbstractFloat}
    point_sets = [convert_arguments(P, quad8) for quad8 in QA]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{Mesh{Tuple{Quadrilateral8_3D{T}}}}, 
        quad8::Quadrilateral8_3D{T}) where {T <: AbstractFloat}
    triangles = triangulate(quad8, 13)
    return convert_arguments(P, triangles)
end

#function convert_arguments(MT::Type{Mesh{Tuple{Quadrilateral8_3D{T}}}},
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
