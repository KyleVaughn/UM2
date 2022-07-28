# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 
# 4th Edition, Chapter 8, Advanced Data Representation


# Function interpolation
# The compiler can't figure out the splatting here without allocations, so we
# write it out explicitly.
# length(w) == length(poly) == N, hence @inbounds is safe
function (poly::Polytope{1, P, N, T})(r) where {P, N, T}
    w = interpolation_weights(Polytope{1, P, N, T}, r)
    p = zero(T)
    @inbounds @simd for i in eachindex(w)
        p += w[i] * poly[i]
    end
    return Point(p)
end

function (poly::Polytope{2, P, N, T})(r, s) where {P, N, T}
    w = interpolation_weights(Polytope{2, P, N, T}, r, s)
    p = zero(T)
    @inbounds @simd for i in eachindex(w)
        p += w[i] * poly[i]
    end
    return Point(p)
end

function (poly::Polytope{3, P, N, T})(r, s, t) where {P, N, T}
    w = interpolation_weights(Polytope{3, P, N, T}, r, s, t)
    p = zero(T)
    @inbounds @simd for i in eachindex(w)
        p += w[i] * poly[i]
    end
    return Point(p)
end

# Shape interpolation
# The compiler can't figure out the splatting here without allocations, so we
# write it out explicitly.
# length(w) == length(poly) == N, hence @inbounds is safe
function (poly::Polytope{1, P, N, Point{D, T}})(r) where {P, N, D, T}
    w = interpolation_weights(Polytope{1, P, N, Point{D, T}}, r)
    p = zero(Vec{D, T})
    @inbounds @simd for i in eachindex(w)
        p += w[i] * coordinates(poly[i])
    end
    return Point(p)
end

function (poly::Polytope{2, P, N, Point{D, T}})(r, s) where {P, N, D, T}
    w = interpolation_weights(Polytope{2, P, N, Point{D, T}}, r, s)
    p = zero(Vec{D, T})
    @inbounds @simd for i in eachindex(w)
        p += w[i] * coordinates(poly[i])
    end
    return Point(p)
end

function (poly::Polytope{3, P, N, Point{3, T}})(r, s, t) where {P, N, T}
    w = interpolation_weights(Polytope{3, P, N, Point{3, T}}, r, s, t)
    p = zero(Vec{3, T})
    @inbounds @simd for i in eachindex(w)
        p += w[i] * coordinates(poly[i])
    end
    return Point(p)
end

# Turn off the JuliaFormatter
#! format: off

# 1-polytope
interpolation_weights(::Type{<:LineSegment},      r) = Vec(1-r, r)
interpolation_weights(::Type{<:QuadraticSegment}, r) = Vec((2r - 1)*( r - 1),
                                                           ( r    )*(2r - 1),
                                                          (-4r    )*( r - 1))
# 2-polytope
interpolation_weights(::Type{<:Triangle},      r, s) = Vec((1 - r - s), r, s)
interpolation_weights(::Type{<:Quadrilateral}, r, s) = Vec((1 - r)*(1 - s), 
                                                           (    r)*(1 - s), 
                                                           (    r)*(    s), 
                                                           (1 - r)*(    s))
interpolation_weights(::Type{<:QuadraticTriangle}, r, s) = Vec((2(1 - r - s) - 1)*(1 - r - s),
                                                               (      r         )*(2r - 1   ),
                                                               (          s     )*(2s - 1   ),
                                                               (     4r         )*(1 - r - s),
                                                               (     4r         )*(        s),
                                                               (         4s     )*(1 - r - s))
function interpolation_weights(::Type{<:QuadraticQuadrilateral}, r, s)
    ξ = 2r - 1; η = 2s - 1
    return Vec((1 - ξ)*(1 - η)*(-ξ - η - 1)/4,
               (1 + ξ)*(1 - η)*( ξ - η - 1)/4,
               (1 + ξ)*(1 + η)*( ξ + η - 1)/4,
               (1 - ξ)*(1 + η)*(-ξ + η - 1)/4,
                          (1 - ξ^2)*(1 - η)/2,
                          (1 - η^2)*(1 + ξ)/2,
                          (1 - ξ^2)*(1 + η)/2,
                          (1 - η^2)*(1 - ξ)/2) 
end

# 3-polytope
interpolation_weights(::Type{<:Tetrahedron}, r, s, t) = Vec((1 - r - s - t), r, s, t)
interpolation_weights(::Type{<:Hexahedron},  r, s, t) = Vec((1 - r)*(1 - s)*(1 - t),
                                                            (    r)*(1 - s)*(1 - t),
                                                            (    r)*(    s)*(1 - t),
                                                            (1 - r)*(    s)*(1 - t),
                                                            (1 - r)*(1 - s)*(    t),
                                                            (    r)*(1 - s)*(    t),
                                                            (    r)*(    s)*(    t),
                                                            (1 - r)*(    s)*(    t))
function interpolation_weights(::Type{<:QuadraticTetrahedron}, r, s, t)
    u = 1 - r - s - t
    return Vec((2u-1)u,
               (2r-1)r,
               (2s-1)s,
               (2t-1)t,
                  4u*r,
                  4r*s,
                  4s*u, 
                  4u*t,
                  4r*t,
                  4s*t)
end
function interpolation_weights(::Type{<:QuadraticHexahedron}, r, s, t)
    ξ = 2r - 1; η = 2s - 1; ζ = 2t - 1
    return Vec((1 - ξ)*(1 - η)*(1 - ζ)*(-2 - ξ - η - ζ)/8,
               (1 + ξ)*(1 - η)*(1 - ζ)*(-2 + ξ - η - ζ)/8,
               (1 + ξ)*(1 + η)*(1 - ζ)*(-2 + ξ + η - ζ)/8,
               (1 - ξ)*(1 + η)*(1 - ζ)*(-2 - ξ + η - ζ)/8,
               (1 - ξ)*(1 - η)*(1 + ζ)*(-2 - ξ - η + ζ)/8,
               (1 + ξ)*(1 - η)*(1 + ζ)*(-2 + ξ - η + ζ)/8,
               (1 + ξ)*(1 + η)*(1 + ζ)*(-2 + ξ + η + ζ)/8,
               (1 - ξ)*(1 + η)*(1 + ζ)*(-2 - ξ + η + ζ)/8,
                          (1 - ξ^2)*(1 - η  )*(1 - ζ  )/4,
                          (1 + ξ  )*(1 - η^2)*(1 - ζ  )/4,
                          (1 - ξ^2)*(1 + η  )*(1 - ζ  )/4,
                          (1 - ξ  )*(1 - η^2)*(1 - ζ  )/4,
                          (1 - ξ^2)*(1 - η  )*(1 + ζ  )/4,
                          (1 + ξ  )*(1 - η^2)*(1 + ζ  )/4,
                          (1 - ξ^2)*(1 + η  )*(1 + ζ  )/4,
                          (1 - ξ  )*(1 - η^2)*(1 + ζ  )/4,
                          (1 - ξ  )*(1 - η  )*(1 - ζ^2)/4,
                          (1 + ξ  )*(1 - η  )*(1 - ζ^2)/4,
                          (1 + ξ  )*(1 + η  )*(1 - ζ^2)/4,
                          (1 - ξ  )*(1 + η  )*(1 - ζ^2)/4)
end
