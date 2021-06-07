import Base: intersect

struct Triangle6{T <: AbstractFloat} <: Face
    points::NTuple{6, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle6(p₁::Point{T}, 
         p₂::Point{T}, 
         p₃::Point{T},
         p₄::Point{T},
         p₅::Point{T},
         p₆::Point{T}
        ) where {T <: AbstractFloat} = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))


# Methods
# -------------------------------------------------------------------------------------------------
function (tri6::Triangle6)(r::T, s::T) where {T <: AbstractFloat}
    weights = [(1 - r - s)*(2(1 - r - s) - 1), 
                                     r*(2r-1),
                                     s*(2s-1),
                               4r*(1 - r - s),
                                         4r*s,
                               4s*(1 - r - s)]
    return sum(weights .* tri6.points) 
end

function derivatives(tri6::Triangle6{T}, r::T, s::T) where {T <: AbstractFloat}
    # Return ( dtri6/dr(r, s), dtri6/ds(r, s) )
    d_dr = (4r + 4s - 3)*tri6.points[1] + 
                (4r - 1)*tri6.points[2] +
           4(1 - 2r - s)*tri6.points[4] +     
                    (4s)*tri6.points[5] +
                   (-4s)*tri6.points[6]

    d_ds = (4r + 4s - 3)*tri6.points[1] + 
                (4s - 1)*tri6.points[3] +
                   (-4r)*tri6.points[4] +
                    (4r)*tri6.points[5] +
           4(1 - r - 2s)*tri6.points[6]     
    return (d_dr, d_ds) 
end

function area(tri6::Triangle6{T}; N::Int64=12) where {T <: AbstractFloat}
    # Numerical integration required. Gauss-Legendre quadrature over a triangle is used.
    # Let T(r,s) be the interpolation function for tri6,
    #                             1 1-r                          N
    # A = ∬ ||∂T/∂r × ∂T/∂s||dA = ∫  ∫ ||∂T/∂r × ∂T/∂s|| ds dr = ∑ wᵢ||∂T/∂r(rᵢ,sᵢ) × ∂T/∂s(rᵢ,sᵢ)||
    #      D                      0  0                          i=1
    #
    # NOTE: for 2D, N = 12 appears to be sufficient. For 3D, N = 79 is preferred.
    w, rs = gauss_legendre_quadrature(tri6, N)
    return sum(w .* norm.([dr × ds for (dr, ds) in [derivatives(tri6, r, s) for (r, s) in rs]]))
end

# triangulate.
# intersect
# in
