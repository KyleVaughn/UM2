# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 
# 4th Edition, Chapter 8, Advanced Data Representation

# Function interpolation
# length(w) == length(poly) == N, hence @inbounds is safe
function (poly::Polytope{K,P,N})(coords...) where {K,P,N}
    w = interpolation_weights(typeof(poly), coords...)
    @inbounds return mapreduce(i->w[i]*poly[i], +, 1:N) 
end

# Shape interpolation
# length(w) == length(poly) == N, hence @inbounds is safe
function (poly::Polytope{K,P,N,T})(coords...) where {K,P,N,T<:Point}
    w = interpolation_weights(typeof(poly), coords...)
    @inbounds return Point(mapreduce(i->w[i]*coordinates(poly[i]), +, 1:N)) 
end

# 1-polytope
interpolation_weights(::Type{<:LineSegment},      r) = Vec(1-r, r)
interpolation_weights(::Type{<:QuadraticSegment}, r) = Vec((2r - 1)*( r - 1),
                                                           ( r    )*(2r - 1),
                                                           (4r    )*( 1 - r))
# 2-polytope
interpolation_weights(::Type{<:Triangle},      r, s) = Vec((1 - r - s), r, s)
interpolation_weights(::Type{<:Quadrilateral}, r, s) = Vec((1 - r)*(1 - s), 
                                                           (    r)*(1 - s), 
                                                           (    r)*(    s), 
                                                           (1 - r)*(    s))
interpolation_weights(::Type{<:QuadraticTriangle}, r, s) = Vec((2(1 - r - s) - 1)*(1 - r - s),
                                                               (      r         )*(2r - 1   ),
                                                               (      s         )*(2s - 1   ),
                                                               (     4r         )*(1 - r - s),
                                                               (     4r         )*(        s),
                                                               (     4s         )*(1 - r - s))
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
