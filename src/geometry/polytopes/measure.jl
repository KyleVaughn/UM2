# Lebesgue measure
export area, arclength, measure, perimeter, volume

arclength(p::Polytope{1}) = measure(p)
area(     p::Polytope{2}) = measure(p)
volume(   p::Polytope{3}) = measure(p)

perimeter(p::Polytope{2}) = mapreduce(measure, +, facets(p))
area(     p::Polytope{3}) = mapreduce(measure, +, facets(p))

measure(l::LineSegment{<:Point}) = norm(l[2]-l[1])

function measure(q::QuadraticSegment{<:Point})
    # The arc length integral may be reduced to an integral over the square root of a
    # quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
    #     1             1
    # L = ∫ ‖𝗾′(r)‖dr = ∫ √(ar² + br + c) dr
    #     0             0
    P₁ = q[1]
    𝘃₁₃ = q[3] - q[1]
    𝘃₁₂ = q[2] - q[1]
    𝘃₂₃ = q[3] - q[2]
    v₁₂ = norm²(𝘃₁₂)
    𝘃₁₄ = (𝘃₁₃ ⋅ 𝘃₁₂)*inv(v₁₂)*𝘃₁₂
    d = norm(𝘃₁₄ - 𝘃₁₃) 
    # If segment is straight
    if d < ϵ_Point
        return √v₁₂ # Distance from P₁ to P₂ 
    else
        𝘂 = 3𝘃₁₃ + 𝘃₂₃
        𝘃 = -2(𝘃₁₃ + 𝘃₂₃)
        a = 4(𝘃 ⋅ 𝘃)
        b = 4(𝘂 ⋅ 𝘃)
        c = 𝘂 ⋅ 𝘂
        
        d = √(a + b + c)
        e = 2a + b
        f = 2√a

        l = (d*e - b*√c) / 4a - (b*b - 4a*c) / (4a*f) * log( (d*f + e) / (√c*f + b) )
        return l
    end
end

measure(tri::Triangle{<:Point}) = norm((tri[2] - tri[1]) × (tri[3] - tri[1]))/2

function measure(poly::Polygon{N, Point{2,T}}) where {N,T}
    # Uses the shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)
    area = zero(T) # Scalar
    @inbounds @simd for i in Base.OneTo(N-1)
        area += coordinates(poly[i]) × coordinates(poly[i + 1])
    end
    @inbounds area += coordinates(poly[N]) × coordinates(poly[1])
    return norm(area)/2
end

function measure(quad::Quadrilateral{Point{3,T}}) where {T}
    # Hexahedron faces are not necessarily planar, hence we use numerical 
    # integration. Gauss-Legendre quadrature over a quadrilateral is used.
    # Let F(r,s) be the interpolation function for the shape. Then,
    #     1 1                          N   N
    # A = ∫ ∫ ‖∂F/∂r × ∂F/∂s‖ ds dr =  ∑   ∑  wᵢwⱼ‖∂F/∂r(rᵢ,sⱼ) × ∂F/∂s(rᵢ,sⱼ)‖
    #     0 0                         i=1 j=1
    # Dispatch on a polynomial order such that the error in computed value when compared to 
    # the reference (BigFloat, P=50), yields error approximately equal to eps(T)
    if T === Float32
        N = 8
    elseif T === Float64
        N = 18
    elseif T === BigFloat
        N = 50 # Would go higher if possible
    else
        error("Unsupported type.")
    end
    weights, points = gauss_quadrature(Val(:legendre), RefLine(), Val(N), T)
    area = zero(T)
    for j in Base.OneTo(N)
        @inbounds @simd for i in Base.OneTo(N) 
            J = jacobian(quad, points[i][1], points[j][1]) 
            area += weights[i]*weights[j]*norm(J[:,1] × J[:,2]) 
        end
    end 
    return area
end

function measure(poly::QuadraticPolygon{N, Point{2,T}}) where {N,T}
    # Let F(r,s) be a parameterization of surface S
    # A = ∬ dS = ∬ ‖∂F/∂r × ∂F/∂s‖dr ds
    #     S      T
    # It can be shown that the area of the quadratic polygon is the area of the base
    # linear shape + the area filled/taken away by the quadratic curves 
    # outside/inside the linear shape. The area under the quadratic edge is 4/3 the 
    # area of the triangle formed by the 3 vertices.
    h = zero(T)
    l = zero(T)
    M = N ÷ 2
    @inbounds @simd for i in Base.OneTo(M-1)
        h += coordinates(poly[i    ]) × coordinates(poly[i + M]) 
        h -= coordinates(poly[i + 1]) × coordinates(poly[i + M])
        l += coordinates(poly[i    ]) × coordinates(poly[i + 1])
    end
    @inbounds h += coordinates(poly[M]) × coordinates(poly[N]) 
    @inbounds h -= coordinates(poly[1]) × coordinates(poly[N])
    @inbounds l += coordinates(poly[M]) × coordinates(poly[1])
    return (4h - l)/6
end
 
# The area integral for 3D quadratic triangles and quadrilaterals appears to have an
# analytic solution, but it involves finding the roots of a quartic polynomial, then 
# integrating over the square root of the factored quartic polynomial. 
# This has a solution in the form of elliptic integrals (See Byrd and Friedman's
# Handbook of Elliptic Integrals for Engineers and Scientists, 2nd edition, 
# equation 251.38), but it's absolutely massive. There may be simplifications after
# the fact that reduce the size of the expression, but for now numerical integration 
# is used.
function measure(tri6::QuadraticTriangle{Point{3,T}}) where {T} 
    # Gauss-Legendre quadrature over a triangle is used.
    # Let F(r,s) be the interpolation function for tri6,
    #            1 1-r                       N                
    # A = ∬ dA = ∫  ∫ ‖∂F/∂r × ∂F/∂s‖ds dr = ∑ wᵢ‖∂F/∂r(rᵢ,sᵢ) × ∂F/∂s(rᵢ,sᵢ)‖
    #     S      0  0                       i=1
    # Dispatch on a polynomial order such that the error in computed value when compared to 
    # the reference (BigFloat, P=20), yields error approximately equal to eps(T)
    if T === Float32
        N = 18
    elseif T === Float64
        N = 20 # Would go higher if possible
    elseif T === BigFloat
        N = 20 # Would go higher if possible
    else
        error("Unsupported type.")
    end
    weights, points = gauss_quadrature(Val(:legendre), RefTriangle(), Val(N), T)
    area = zero(T)
    @inbounds @simd for i in eachindex(weights)
        J = jacobian(tri6, points[i][1], points[i][2])
        area += weights[i] * norm(J[:,1] × J[:,2]) 
    end
    return area
end

function measure(quad8::QuadraticQuadrilateral{Point{3,T}}) where {T}
    # Let F(r,s) be the interpolation function for the shape. Then,
    #     1 1                          N   N
    # A = ∫ ∫ ‖∂F/∂r × ∂F/∂s‖ ds dr =  ∑   ∑  wᵢwⱼ‖∂F/∂r(rᵢ,sⱼ) × ∂F/∂s(rᵢ,sⱼ)‖
    #     0 0                         i=1 j=1
    # Dispatch on a polynomial order such that the error in computed value when compared to 
    # the reference (BigFloat, P=50), yields error approximately equal to eps(T)
    if T === Float32
        N = 15
    elseif T === Float64
        N = 38
    elseif T === BigFloat
        N = 50 # Would go higher if possible
    else
        error("Unsupported type.")
    end
    weights, points = gauss_quadrature(Val(:legendre), RefLine(), Val(N),T)
    area = zero(T)
    for j in Base.OneTo(N)
        @inbounds @simd for i in Base.OneTo(N) 
            J = jacobian(quad8, points[i][1], points[j][1]) 
            area += weights[i]*weights[j]*norm(J[:,1] × J[:,2]) 
        end
    end 
    return area
end
