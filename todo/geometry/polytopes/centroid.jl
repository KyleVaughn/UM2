export centroid

# (https://en.wikipedia.org/wiki/Centroid#Of_a_polygon)
function centroid(poly::Polygon{N, Point{2, T}}) where {N, T}
    a = zero(T)
    c = Vec{2, T}(0, 0)
    vec1 = coordinates(poly[1])
    for i in Base.OneTo(N - 1)
        vec2 = coordinates(poly[i + 1])
        subarea = vec1 × vec2
        c += subarea * (vec1 + vec2)
        a += subarea
        vec1 = vec2
    end
    vec2 = coordinates(poly[1])
    subarea = vec1 × vec2
    c += subarea * (vec1 + vec2)
    a += subarea
    return Point(c / (3a))
end

centroid(tri6::QuadraticTriangle{<:Point{2}}) = centroid(tri6, Val(4))
function centroid(tri6::QuadraticTriangle{Point{D, T}}, ::Val{N}) where {D, T, N}
    # Gauss-Legendre quadrature over a triangle is used.
    # Let f(r,s) be the interpolation function for tri6,
    #            1 1-r                       N
    # A = ∬ dA = ∫  ∫ ‖∂f/∂r × ∂f/∂s‖ds dr = ∑ wᵢ‖∂f/∂r(rᵢ,sᵢ) × ∂f/∂s(rᵢ,sᵢ)‖
    #     S      0  0                       i=1
    #                  1  N
    # C = (∬ 𝘅 dA)/A = -  ∑ 𝘅 wᵢ‖∂f/∂r(rᵢ,sᵢ) × ∂f/∂s(rᵢ,sᵢ)‖
    #      S           A i=1
    wts, pts = gauss_quadrature(LegendreType(), RefTriangle(), Val(N), T)
    a = zero(T)
    c = @SVector zeros(T, D)
    @inbounds @simd for i in Base.OneTo(length(wts))
        J = jacobian(tri6, pts[i]...)
        weighted_val = wts[i] * norm(view(J, :, 1) × view(J, :, 2))
        c += weighted_val * coordinates(tri6(pts[i]...))
        a += weighted_val
    end
    return Point(c / a)
end

centroid(quad8::QuadraticQuadrilateral{<:Point{2}}) = centroid(quad8, Val(3))
function centroid(quad8::QuadraticQuadrilateral{Point{D, T}}, ::Val{N}) where {D, T, N}
    # Gauss-Legendre quadrature over a quadrilateral is used.
    # Let f(r,s) be the interpolation function for quad8,
    #            1  1                        N   N
    # A = ∬ dA = ∫  ∫ ‖∂f/∂r × ∂f/∂s‖ds dr = ∑   ∑ wᵢwⱼ‖∂f/∂r(rᵢ,sⱼ) × ∂f/∂s(rᵢ,sⱼ)‖
    #     S      0  0                       j=1 i=1
    #                  1  N   N
    # C = (∬ 𝘅 dA)/A = -  ∑   ∑ 𝘅ᵢⱼwᵢwⱼ‖∂f/∂r(rᵢ,sⱼ) × ∂f/∂s(rᵢ,sⱼ)‖
    #      S           A j=1 i=1
    wts, pts = gauss_quadrature(LegendreType(), RefSquare(), Val(N), T)
    a = zero(T)
    c = @SVector zeros(T, D)
    @inbounds @simd for i in Base.OneTo(length(wts))
        J = jacobian(quad8, pts[i]...)
        weighted_val = wts[i] * norm(view(J, :, 1) × view(J, :, 2))
        c += weighted_val * coordinates(quad8(pts[i]...))
        a += weighted_val
    end
    return Point(c / a)
end
