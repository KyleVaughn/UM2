@inline measure(aab::AABox2D) = Δx(aab) * Δy(aab)
@inline measure(aab::AABox3D) = Δx(aab) * Δy(aab) * Δz(aab)
@inline measure(l::LineSegment) = distance(l.𝘅₁.coord, l.𝘅₁.coord + l.𝘂)
function measure(q::QuadraticSegment)
    # The arc length integral may be reduced to an integral over the square root of a
    # quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
    #     1             1
    # L = ∫ ‖𝗾′(r)‖dr = ∫ √(ar² + br + c) dr
    #     0             0
    if isstraight(q)
        return distance(q.𝘅₁, q.𝘅₂)
    else
        𝘂 = q.𝘂
        𝘃 = q.𝘃
        a = 4(𝘂 ⋅ 𝘂)
        b = 4(𝘂 ⋅ 𝘃)
        c = 𝘃 ⋅ 𝘃
        # Compiler seems to catch the reused sqrt quantities for common subexpression
        # elimination, or computation is as quick as storage in a variable, so we
        # leave the sqrts for readability
        l = ((2a + b)√(a + b + c) - b√c)/4a -
            (b^2 - 4a*c)/((2√a)^3)*log((2√a√(a + b + c) + (2a + b))/(2√a√c + b))
        return l
    end
end

# Area
# ---------------------------------------------------------------------------------------------
# Uses the shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)
function area(poly::Polygon{N, 2, T}) where {N, T}
    a = zero(T) # Scalar
    for i ∈ 1:N-1
        a += poly[i] × poly[i + 1]
    end
    a += poly[N] × poly[1]
    return norm(a)/2
end
# Not necessarily planar
#function area(poly::Polygon{N, 3, T}) where {N, T}
#    a = zero(SVector{3, T}) # Vector
#    for i ∈ 1:N-1
#        a += (poly[i] × poly[i + 1])
#    end
#    a += poly[N] × poly[1]
#    return norm(a)/2
#end
#

area(tri::Triangle2D) = norm((tri[2] - tri[1]) × (tri[3] - tri[1]))/2
area(tri::Triangle3D) = norm((tri[2] - tri[1]) × (tri[3] - tri[1]))/2

# Methods
# ---------------------------------------------------------------------------------------------
# area
#
# In general:
# Let 𝗳(r,s) be a parameterization of surface S
# A = ∬ dS = ∬ ‖∂𝗳/∂r × ∂𝗳/∂s‖dr ds
#     S      T
function area(poly::QuadraticPolygon{N,2,T}) where {N,T}
    # It can be shown that the area of the quadratic polygon is the area of the base
    # linear shape + the area filled by the quadratic curves outside/inside the linear
    # shape. The area of the quadratic edge is 4/3 the area of the triangle formed by the
    # 3 vertices.
    q = zero(T)
    l = zero(T)
    M = N ÷ 2
    for i ∈ 1:M-1
        q += poly[i    ] × poly[i + M]
        q -= poly[i + 1] × poly[i + M]
        l += poly[i] × poly[i + 1]
    end
    q += poly[M] × poly[N]
    q -= poly[1] × poly[N]
    l += poly[M] × poly[1]
    return (4q - l)/6
end

# The area integral for 3D quadratic triangles and quadrilaterals appears to have an
# analytic solution, but it involves finding the roots of a quartic polynomial, then 
# integrating over the square root of the factored quartic polynomial. 
# This has a solution in the form of elliptic integrals (See Byrd and Friedman's
# Handbook of Elliptic Integrals for Engineers and Scientists, 2nd edition, 
# equation 251.38), but it's absolutely massive. There may be simplifications after
# the fact that reduce the size of the expression, but for now numerical integration is 
# quicker.
function area(quad8::QuadraticQuadrilateral3D{T}, ::Val{N}) where {T, N}
    # Gauss-Legendre quadrature over a quadrilateral is used.
    # Let Q(r,s) be the interpolation function for quad8,
    #     1 1                          N   N
    # A = ∫ ∫ ‖∂Q/∂r × ∂Q/∂s‖ ds dr =  ∑   ∑  wᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
    #     0 0                         i=1 j=1
    w, r = gauss_legendre_quadrature(T, Val(N))
    A = zero(T)
    for j ∈ 1:N, i ∈ 1:N 
        J = 𝗝(quad8, r[i], r[j]) 
        A += w[i]*w[j]*norm(view(J, :, 1) × view(J, :, 2)) 
    end 
    return A
end
function area(tri6::QuadraticTriangle3D{T}, ::Val{N}) where {T, N} 
    # Gauss-Legendre quadrature over a triangle is used.
    # Let F(r,s) be the interpolation function for tri6,
    #            1 1-r                       N                
    # A = ∬ dA = ∫  ∫ ‖∂F/∂r × ∂F/∂s‖ds dr = ∑ wᵢ‖∂F/∂r(rᵢ,sᵢ) × ∂F/∂s(rᵢ,sᵢ)‖
    #     S      0  0                       i=1
    w, r, s = gauss_legendre_quadrature(tri6, Val(N))
    A = zero(T)
    for i ∈ 1:N
        J = 𝗝(tri6, r[i], s[i])
        A += w[i] * norm(view(J, :, 1) × view(J, :, 2)) 
    end
    return A
end

# Return the area of face id
function area(id, mesh::UnstructuredMesh)
    return area(materialize_face(id, mesh))
end

# Return the area of the entire mesh
function area(mesh::UnstructuredMesh)
    # use sum
    return mapreduce(x->area(x, mesh), +, 1:length(mesh.faces))
end

# Return the area of a face set
function area(face_set::BitSet, mesh::UnstructuredMesh)
    # use sum
    return mapreduce(x->area(x, mesh), +, face_set)
end

# Return the area of a face set by name
function area(set_name::String, mesh::UnstructuredMesh)
    return area(mesh.face_sets[set_name], mesh)
end
