"""
    triangulate(poly::Polygon{N, 2, T}) where {N, T}

Return an SVector of `N`-2 triangles that partition the `Polygon`. 
Generated using fan triangulation.
"""
function triangulate(poly::Polygon{N, 2, T}) where {N, T}
    triangles = MVector{N-2, Triangle{2, T}}(undef)
    for i = 1:N-2
        triangles[i] = Triangle(poly[1], poly[i+1], poly[i+2])
    end 
    return SVector(triangles.data)
end

"""
    triangulate(poly::Polygon{N, 2, BigFloat}) where {N}

Return a Vector of `N`-2 triangles that partition the `Polygon`. 
Generated using fan triangulation.
"""
function triangulate(poly::Polygon{N, 2, BigFloat}) where {N}
    triangles = Vector{Triangle{2, BigFloat}}(undef, N-2)
    for i = 1:N-2
        triangles[i] = Triangle(poly[1], poly[i+1], poly[i+2])
    end 
    return triangles
end

"""
    triangulate(quad::Quadrilateral3D{T}, ::Val{N}) where {N, T}

Return an SVector of (2`N`+2)^2 triangles that partition the `Quadrilateral3D`. 
"""
function triangulate(quad::Quadrilateral3D{T}, ::Val{N}) where {T, N}
    # N is the number of divisions of each edge
    N1 = N + 1
    inv_N1 = T(1)/N1
    triangles = MVector{2N1^2, Triangle3D{T}}(undef)
    if N === 0
        triangles[1] = Triangle(quad[1], quad[2], quad[3])
        triangles[2] = Triangle(quad[3], quad[4], quad[1])
    else
        for j = 0:N
            s₀ = inv_N1*j
            s₁ = inv_N1*(j + 1)
            for i = 0:N
                r₀ = inv_N1*i
                r₁ = inv_N1*(i + 1)
                v₀₀ = quad(r₀, s₀)
                v₁₀ = quad(r₁, s₀)
                v₀₁ = quad(r₀, s₁)
                v₁₁ = quad(r₁, s₁)
                triangles[2*(N1*j + i) + 1] = Triangle(v₀₀, v₁₀, v₀₁)
                triangles[2*(N1*j + i) + 2] = Triangle(v₀₁, v₁₀, v₁₁)
            end
        end
    end
    return SVector(triangles.data)
end

"""
    triangulate(quad::Quadrilateral3D{BigFloat}, ::Val{N}) where {N}

Return a Vector of (2`N`+2)^2 triangles that partition the `Quadrilateral3D`. 
"""
function triangulate(quad::Quadrilateral3D{BigFloat}, ::Val{N}) where {N}
    # N is the number of divisions of each edge
    N1 = N + 1
    inv_N1 = BigFloat(1)/N1
    triangles = Vector{Triangle3D{BigFloat}}(undef, 2N1^2)
    if N === 0
        triangles[1] = Triangle(quad[1], quad[2], quad[3])
        triangles[2] = Triangle(quad[3], quad[4], quad[1])
    else
        for j = 0:N
            s₀ = inv_N1*j
            s₁ = inv_N1*(j + 1)
            for i = 0:N
                r₀ = inv_N1*i
                r₁ = inv_N1*(i + 1)
                v₀₀ = quad(r₀, s₀)
                v₁₀ = quad(r₁, s₀)
                v₀₁ = quad(r₀, s₁)
                v₁₁ = quad(r₁, s₁)
                triangles[2*(N1*j + i) + 1] = Triangle(v₀₀, v₁₀, v₀₁)
                triangles[2*(N1*j + i) + 2] = Triangle(v₀₁, v₁₀, v₁₁)
            end
        end
    end
    return triangles
end

"""
    triangulate(tri6::QuadraticTriangle{Dim, T}, ::Val{N}) where {Dim, T, N}

Return an SVector of (2`N`+2)^2 triangles that partition the `QuadraticTriangle`. 
"""
function triangulate(tri6::QuadraticTriangle{Dim, T}, ::Val{N}) where {Dim, T, N}
    N1 = N + 1
    inv_N1 = T(1)/N1
    triangles = MVector{N1^2, Triangle{Dim, T}}(undef)
    if N === 0
        triangles[1] = Triangle(tri6[1], tri6[2], tri6[3])
    else
        i = 1
        for s ∈ 1:N
            s₋₁ = inv_N1*(s-1)
            s₀ = inv_N1*s
            s₁ = inv_N1*(s + 1)
            for r ∈ 0:N-s
                r₀ = inv_N1*r
                r₁ = inv_N1*(r + 1)
                v₀₀ = tri6(r₀, s₀)
                v₁₀ = tri6(r₁, s₀ )
                v₀₁ = tri6(r₀, s₁)
                v₁₋₁ = tri6(r₁, s₋₁)
                triangles[i]   = Triangle(v₀₀, v₁₀ , v₀₁)
                triangles[i+1] = Triangle(v₀₀, v₁₋₁, v₁₀)
                i += 2
            end
        end
        j = N1*N + 1
        s₀ = zero(T)
        s₁ = inv_N1
        for r ∈ 0:N
            r₀ = inv_N1*r
            r₁ = inv_N1*(r + 1)
            triangles[j] = Triangle(tri6(r₀, s₀), tri6(r₁, s₀), tri6(r₀, s₁))
            j += 1
        end
    end
    return SVector(triangles.data)
end

"""
    triangulate(tri6::QuadraticTriangle{Dim, BigFloat}, ::Val{N}) where {Dim, N}

Return a Vector of (2`N`+2)^2 triangles that partition the `QuadraticTriangle`. 
"""
function triangulate(tri6::QuadraticTriangle{Dim, BigFloat}, ::Val{N}) where {Dim, N}
    N1 = N + 1
    inv_N1 = BigFloat(1)/N1
    triangles = Vector{Triangle{Dim, BigFloat}}(undef, N1^2)
    if N === 0
        triangles[1] = Triangle(tri6[1], tri6[2], tri6[3])
    else
        i = 1
        for s ∈ 1:N
            s₋₁ = inv_N1*(s-1)
            s₀ = inv_N1*s
            s₁ = inv_N1*(s + 1)
            for r ∈ 0:N-s
                r₀ = inv_N1*r
                r₁ = inv_N1*(r + 1)
                v₀₀ = tri6(r₀, s₀)
                v₁₀ = tri6(r₁, s₀ )
                v₀₁ = tri6(r₀, s₁)
                v₁₋₁ = tri6(r₁, s₋₁)
                triangles[i]   = Triangle(v₀₀, v₁₀ , v₀₁)
                triangles[i+1] = Triangle(v₀₀, v₁₋₁, v₁₀)
                i += 2
            end
        end
        j = N1*N + 1
        s₀ = zero(BigFloat)
        s₁ = inv_N1
        for r ∈ 0:N
            r₀ = inv_N1*r
            r₁ = inv_N1*(r + 1)
            triangles[j] = Triangle(tri6(r₀, s₀), tri6(r₁, s₀), tri6(r₀, s₁))
            j += 1
        end
    end
    return triangles
end

"""
    triangulate(quad::QuadraticQuadrilateral{Dim, T}, ::Val{N}) where {Dim, T, N}

Return an SVector of (2`N`+2)^2 triangles that partition the `QuadraticQuadrilateral`. 
"""
function triangulate(quad8::QuadraticQuadrilateral{Dim, T}, ::Val{N}) where {Dim, T, N}
    # N is the number of divisions of each edge
    N1 = N + 1
    inv_N1 = T(1)/N1
    triangles = MVector{2N1^2, Triangle{Dim, T}}(undef)
    if N === 0
        triangles[1] = Triangle(quad8[1], quad8[2], quad8[3])
        triangles[2] = Triangle(quad8[3], quad8[4], quad8[1])
    else
        for j = 0:N
            s₀ = inv_N1*j
            s₁ = inv_N1*(j + 1)
            for i = 0:N
                r₀ = inv_N1*i
                r₁ = inv_N1*(i + 1)
                v₀₀ = quad8(r₀, s₀)
                v₁₀ = quad8(r₁, s₀)
                v₀₁ = quad8(r₀, s₁)
                v₁₁ = quad8(r₁, s₁)
                triangles[2*(N1*j + i) + 1] = Triangle(v₀₀, v₁₀, v₀₁)
                triangles[2*(N1*j + i) + 2] = Triangle(v₀₁, v₁₀, v₁₁)
            end
        end
    end
    return SVector(triangles.data)
end

"""
    triangulate(quad::QuadraticQuadrilateral{Dim, BigFloat}, ::Val{N}) where {Dim, N}

Return a Vector of (2`N`+2)^2 triangles that partition the `QuadraticQuadrilateral`. 
"""
function triangulate(quad8::QuadraticQuadrilateral{Dim, BigFloat}, 
                     ::Val{N}) where {Dim, N}
    # N is the number of divisions of each edge
    N1 = N + 1
    inv_N1 = BigFloat(1)/N1
    triangles = Vector{Triangle{Dim, BigFloat}}(undef, 2N1^2)
    if N === 0
        triangles[1] = Triangle(quad8[1], quad8[2], quad8[3])
        triangles[2] = Triangle(quad8[3], quad8[4], quad8[1])
    else
        for j = 0:N
            s₀ = inv_N1*j
            s₁ = inv_N1*(j + 1)
            for i = 0:N
                r₀ = inv_N1*i
                r₁ = inv_N1*(i + 1)
                v₀₀ = quad8(r₀, s₀)
                v₁₀ = quad8(r₁, s₀)
                v₀₁ = quad8(r₀, s₁)
                v₁₁ = quad8(r₁, s₁)
                triangles[2*(N1*j + i) + 1] = Triangle(v₀₀, v₁₀, v₀₁)
                triangles[2*(N1*j + i) + 2] = Triangle(v₀₁, v₁₀, v₁₁)
            end
        end
    end
    return triangles
end
