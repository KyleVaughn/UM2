export QuadraticTriangle,
       QuadraticTriangle2,
       QuadraticTriangle2f,
       QuadraticTriangle2d

export interpolate_quadratic_triangle,
       jacobian,
       quadratic_triangle_jacobian,
       area,
       centroid,
       edge,
       edges,
       bounding_box,
       triangulate

# QUADRATIC TRIANGLE 
# -----------------------------------------------------------------------------
#
# A quadratic triangle represented by 6 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct QuadraticTriangle{D, T}
    vertices::Vec{6, Point{D, T}}
end

# -- Type aliases --

const QuadraticTriangle2  = QuadraticTriangle{2}
const QuadraticTriangle2f = QuadraticTriangle2{Float32}
const QuadraticTriangle2d = QuadraticTriangle2{Float64}

# -- Base --

Base.getindex(T::QuadraticTriangle, i) = T.vertices[i]
Base.broadcastable(T::QuadraticTriangle) = Ref(T)

# -- Constructors --

function QuadraticTriangle(
        P1::Point{D, T}, 
        P2::Point{D, T}, 
        P3::Point{D, T},
        P4::Point{D, T},
        P5::Point{D, T},
        P6::Point{D, T}) where {D, T}
    return QuadraticTriangle{D, T}(Vec(P1, P2, P3, P4, P5, P6))
end

# -- Interpolation --

function interpolate_quadratic_triangle(
        P1::T, P2::T, P3::T, P4::T, P5::T, P6::T, r, s) where {T}
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s)) * P1 +    
           (          r           * ( 2 * r - 1)) * P2 +    
           (              s       * ( 2 * s - 1)) * P3 +    
           (      4 * r           * ( 1 - r - s)) * P4 +    
           (      4 * r           *           s ) * P5 +    
           (          4 * s       * ( 1 - r - s)) * P6
end

function interpolate_quadratic_triangle(vertices::Vec{6}, r, s)
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s)) * vertices[1] +    
           (          r           * ( 2 * r - 1)) * vertices[2] +    
           (              s       * ( 2 * s - 1)) * vertices[3] +    
           (      4 * r           * ( 1 - r - s)) * vertices[4] +    
           (      4 * r           *           s ) * vertices[5] +    
           (          4 * s       * ( 1 - r - s)) * vertices[6]
end

function (T::QuadraticTriangle{D, F})(r::F, s::F) where {D, F}
    return interpolate_quadratic_triangle(T.vertices, r, s)
end

# -- Jacobian --

function quadratic_triangle_jacobian(
        P1::T, P2::T, P3::T, P4::T, P5::T, P6::T, r, s) where {T}
    ∂r = (4 * r + 4 * s - 3) * (P1 - P4) +    
         (4 * r         - 1) * (P2 - P4) +    
         (        4 * s    ) * (P5 - P6)
    
    ∂s = (4 * r + 4 * s - 3) * (P1 - P6) +
         (        4 * s - 1) * (P3 - P6) +
         (4 * r            ) * (P5 - P4)
    
    return Mat(∂r, ∂s)
end

function quadratic_triangle_jacobian(vertices::Vec{6}, r, s)
    ∂r = (4 * r + 4 * s - 3) * (vertices[1] - vertices[4]) +    
         (4 * r         - 1) * (vertices[2] - vertices[4]) +    
         (        4 * s    ) * (vertices[5] - vertices[6])
    
    ∂s = (4 * r + 4 * s - 3) * (vertices[1] - vertices[6]) +
         (        4 * s - 1) * (vertices[3] - vertices[6]) +
         (4 * r            ) * (vertices[5] - vertices[4])
    
    return Mat(∂r, ∂s)
end

function jacobian(T::QuadraticTriangle{D, F}, r::F, s::F) where {D, F}
    return quadratic_triangle_jacobian(T.vertices, r, s)
end

# -- Measure --

function area(T::QuadraticTriangle{2, F}) where {F}
    # The area enclosed by the 3 quadratic edges + the area enclosed
    # by the triangle (P1, P2, P3)
    edge_area = F(2//3) * ((T[4] - T[1]) × (T[2] - T[1])  +
                           (T[5] - T[2]) × (T[3] - T[2])  +
                           (T[6] - T[3]) × (T[1] - T[3])) 
    tri_area = ((T[2] - T[1]) × (T[3] - T[1])) / 2 
    return edge_area + tri_area
end

# -- Centroid --

function centroid(T::QuadraticTriangle{2, F}) where {F}
    # By geometric decomposition into a triangle and the 3 areas
    # enclosed by the quadratic edges.
    aₜ = triangle_area(T[1], T[2], T[3])
    Cₜ = triangle_centroid(T[1], T[2], T[3])
    a₁ = area_enclosed_by_quadratic_segment(T[1], T[2], T[4])
    a₂ = area_enclosed_by_quadratic_segment(T[2], T[3], T[5])
    a₃ = area_enclosed_by_quadratic_segment(T[3], T[1], T[6])
    C₁ = centroid_of_area_enclosed_by_quadratic_segment(T[1], T[2], T[4])
    C₂ = centroid_of_area_enclosed_by_quadratic_segment(T[2], T[3], T[5])
    C₃ = centroid_of_area_enclosed_by_quadratic_segment(T[3], T[1], T[6])
    return (aₜ*Cₜ + a₁ * C₁ + a₂ * C₂ + a₃ * C₃) / (aₜ + a₁ + a₂ + a₃)
end

# -- Edges --

function edge(i::Integer, t::QuadraticTriangle)
    # Assumes 1 ≤ i ≤ 3.
    if i < 3
        return QuadraticSegment(T[i], T[i+1], T[i+3])
    else
        return QuadraticSegment(T[3], T[1], T[6])
    end
end

edges(T::QuadraticTriangle) = (edge(i, T) for i in 1:3)

# -- Bounding box --

function bounding_box(T::QuadraticTriangle)
    return bounding_box(edge(1, Q)) ∪
           bounding_box(edge(2, Q)) ∪
           bounding_box(edge(3, Q))
end

# -- In --    
      
Base.in(P::Point{2}, T::QuadraticTriangle{2}) = all(edge -> isleft(P, edge), edges(T))

# -- Triangulation --

# N is the number of segments to divide each edge into.
# Return a Vector of the N^2 triangles that approximately partition 
# the quadratic triangle.
function triangulate(T::QuadraticTriangle{D, F}, N::Integer) where {D, F}
    # Walk up the triangle in parametric coordinates (r as fast variable,
    # s as slow variable).
    # r is incremented along each row, forming two triangles (1,2,3) and
    # (3,2,4), at each step, until the last r value, which is a single triangle.
    # 3 --- 4        3 
    # | \   |        | \   
    # |   \ |        |   \  
    # 1 --- 2 ...... 1 --- 2 
    triangles = Vector{Triangle{D, F}}(undef, N^2)
    Δ = F(1) / N # Δ for r and s
    s1 = F(0)
    ntri = 0
    for s = 0:(N - 1)
        r1 = F(0)
        s2 = s1 + Δ
        P1 = T(r1, s1)
        P3 = T(r1, s2)
        for r = 0:(N - 2 - s)
            r2 = r1 + Δ
            P2 = T(r2, s1)
            P4 = T(r2, s2)
            triangles[ntri + 2 * r + 1] = Triangle(P1, P2, P3)
            triangles[ntri + 2 * r + 2] = Triangle(P3, P2, P4)
            r1 = r2
            P1 = P2
            P3 = P4
        end
        ntri += 2*(N - s) - 1
        triangles[ntri] = Triangle(P1, t(r1 + Δ, s1), P3)
        s1 = s2
    end
    return triangles
end

# -- IO --

function Base.show(io::IO, T::QuadraticTriangle{D, F}) where {D, F}
    type_char = '?'
    if F === Float32
        type_char = 'f'
    elseif F === Float64
        type_char = 'd'
    end
    print(io, "QuadraticTriangle", D, type_char, '(', 
        T.vertices[1], ", ", 
        T.vertices[2], ", ", 
        T.vertices[3], ", ",
        T.vertices[4], ", ",
        T.vertices[5], ", ",
        T.vertices[6], ')')
end
