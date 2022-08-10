export triangulate

"""
    triangulate(poly::Polygon{N,Point{2,T}}) where {N,T}

Return a Vec of the `N`-2 triangles that partition the 2D `Polygon`. 

Generated using fan triangulation, which assumes the `Polygon` is convex.
"""
function triangulate(p::Polygon{N, Point{2, T}}) where {N, T}
    return Vec{N - 2, Triangle{Point{2, T}}}(ntuple(i -> Triangle(p[1], p[i + 1], p[i + 2]),
                                                    Val(N - 2)))
end

"""
    triangulate(quad::Quadrilateral{Point{3,T}}, ::Val{N}) where {N,T}

Return a Vec of the (2`N`+2)^2 triangles that approximately partition the `Quadrilateral`. 
"""
@generated function triangulate(quad::Quadrilateral{Point{3, T}}, ::Val{N}) where {T, N}
    # N is the number of divisions of each edge
    N1 = N + 1
    inv_N1 = T(1) / N1
    if N === 0
        exprs = [:(Triangle(quad[1], quad[2], quad[3])),
            :(Triangle(quad[3], quad[4], quad[1])),
        ]
    else
        exprs = Expr[]
        for j in 0:N
            s₀ = inv_N1 * j
            s₁ = inv_N1 * (j + 1)
            for i in 0:N
                r₀ = inv_N1 * i
                r₁ = inv_N1 * (i + 1)
                push!(exprs, :(Triangle(quad($r₀, $s₀), quad($r₁, $s₀), quad($r₀, $s₁))))
                push!(exprs, :(Triangle(quad($r₀, $s₁), quad($r₁, $s₀), quad($r₁, $s₁))))
            end
        end
    end
    return quote
        return Vec{$(2N1^2), Triangle{Point{3, $T}}}(tuple($(exprs...)))
    end
end

"""
    triangulate(tri6::QuadraticTriangle{Point{D,T}}, ::Val{N}) where {D,T,N}

Return a Vec of (2`N`+2)^2 triangles that approximately partition the `QuadraticTriangle`. 
"""
@generated function triangulate(tri6::QuadraticTriangle{Point{D, T}},
                                ::Val{N}) where {D, T, N}
    # N is the number of divisions of each edge
    N1 = N + 1
    inv_N1 = T(1) / N1
    if N === 0
        exprs = [:(Triangle(tri6[1], tri6[2], tri6[3]))]
    else
        exprs = Expr[]
        for s in 1:N
            s₋₁ = inv_N1 * (s - 1)
            s₀  = inv_N1 * s
            s₁  = inv_N1 * (s + 1)
            for r in 0:(N - s)
                r₀ = inv_N1 * r
                r₁ = inv_N1 * (r + 1)
                push!(exprs, :(Triangle(tri6($r₀, $s₀), tri6($r₁, $s₀), tri6($r₀, $s₁))))
                push!(exprs, :(Triangle(tri6($r₀, $s₀), tri6($r₁, $s₋₁), tri6($r₁, $s₀))))
            end
        end
        j = N1 * N + 1
        s₀ = zero(T)
        s₁ = inv_N1
        for r in 0:N
            r₀ = inv_N1 * r
            r₁ = inv_N1 * (r + 1)
            push!(exprs, :(Triangle(tri6($r₀, $s₀), tri6($r₁, $s₀), tri6($r₀, $s₁))))
        end
    end
    return quote
        return Vec{$(N1^2), Triangle{Point{$D, $T}}}(tuple($(exprs...)))
    end
end

"""
    triangulate(quad::QuadraticQuadrilateral{Point{D,T}}, ::Val{N}) where {D,T,N}

Return a Vec of (2`N`+2)^2 triangles that approximately partition the `QuadraticQuadrilateral`. 
"""
@generated function triangulate(quad8::QuadraticQuadrilateral{Point{D, T}},
                                ::Val{N}) where {D, T, N}
    # N is the number of divisions of each edge
    N1 = N + 1
    inv_N1 = T(1) / N1
    if N === 0
        exprs = [:(Triangle(quad8[1], quad8[2], quad8[3])),
            :(Triangle(quad8[3], quad8[4], quad8[1])),
        ]
    else
        exprs = Expr[]
        for j in 0:N
            s₀ = inv_N1 * j
            s₁ = inv_N1 * (j + 1)
            for i in 0:N
                r₀ = inv_N1 * i
                r₁ = inv_N1 * (i + 1)
                push!(exprs, :(Triangle(quad8($r₀, $s₀), quad8($r₁, $s₀), quad8($r₀, $s₁))))
                push!(exprs, :(Triangle(quad8($r₀, $s₁), quad8($r₁, $s₀), quad8($r₁, $s₁))))
            end
        end
    end
    return quote
        return Vec{$(2N1^2), Triangle{Point{3, $T}}}(tuple($(exprs...)))
    end
end
