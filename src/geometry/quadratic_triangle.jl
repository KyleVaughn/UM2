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

struct QuadraticTriangle{D, T} <: QuadraticPolygon{D, T}
    vertices::Vec{6, Point{D, T}}
end

# -- Type aliases --

const QuadraticTriangle2  = QuadraticTriangle{2}
const QuadraticTriangle2f = QuadraticTriangle2{Float32}
const QuadraticTriangle2d = QuadraticTriangle2{Float64}

# -- Base --

Base.getindex(t::QuadraticTriangle, i) = t.vertices[i]
Base.broadcastable(t::QuadraticTriangle) = Ref(t)

# -- Constructors --

function QuadraticTriangle(
        p1::Point{D, T}, 
        p2::Point{D, T}, 
        p3::Point{D, T},
        p4::Point{D, T},
        p5::Point{D, T},
        p6::Point{D, T}) where {D, T}
    return QuadraticTriangle{D, T}(Vec(p1, p2, p3, p4, p5, p6))
end

# -- Interpolation --

function interpolate_quadratic_triangle(
        p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, r, s) where {T}
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s)) * p1 +    
           (          r           * ( 2 * r - 1)) * p2 +    
           (              s       * ( 2 * s - 1)) * p3 +    
           (      4 * r           * ( 1 - r - s)) * p4 +    
           (      4 * r           *           s ) * p5 +    
           (          4 * s       * ( 1 - r - s)) * p6
end

function interpolate_quadratic_triangle(vertices::Vec{6}, r, s)
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s)) * vertices[1] +    
           (          r           * ( 2 * r - 1)) * vertices[2] +    
           (              s       * ( 2 * s - 1)) * vertices[3] +    
           (      4 * r           * ( 1 - r - s)) * vertices[4] +    
           (      4 * r           *           s ) * vertices[5] +    
           (          4 * s       * ( 1 - r - s)) * vertices[6]
end

function (t::QuadraticTriangle{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadratic_triangle(t.vertices, r, s)
end

# -- Jacobian --

function quadratic_triangle_jacobian(
        p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, r, s) where {T}
    ∂r = (4 * r + 4 * s - 3) * (p1 - p4) +    
         (4 * r         - 1) * (p2 - p4) +    
         (        4 * s    ) * (p5 - p6)
    
    ∂s = (4 * r + 4 * s - 3) * (p1 - p6) +
         (        4 * s - 1) * (p3 - p6) +
         (4 * r            ) * (p5 - p4)
    
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

function jacobian(t::QuadraticTriangle{D, T}, r::T, s::T) where {D, T}
    return quadratic_triangle_jacobian(t.vertices, r, s)
end

# -- Measure --

function area(t::QuadraticTriangle{2, T}) where {T}
    # The area enclosed by the 3 quadratic edges + the area enclosed
    # by the triangle (p1, p2, p3)
    edge_area = T(2//3) * ((t[4] - t[1]) × (t[2] - t[1])  +
                           (t[5] - t[2]) × (t[3] - t[2])  +
                           (t[6] - t[3]) × (t[1] - t[3])) 
    tri_area = ((t[2] - t[1]) × (t[3] - t[1])) / 2 
    return edge_area + tri_area
end

# -- Centroid --

function centroid(t::QuadraticTriangle{2, T}) where {T}
    # By geometric decomposition into a triangle and the 3 areas
    # enclosed by the quadratic edges.
    Aₜ = triangle_area(t[1], t[2], t[3])
    Cₜ = triangle_centroid(t[1], t[2], t[3])
    A₁ = area_enclosed_by_quadratic_segment(t[1], t[2], t[4])
    C₁ = centroid_of_area_enclosed_by_quadratic_segment(t[1], t[2], t[4])
    A₂ = area_enclosed_by_quadratic_segment(t[2], t[3], t[5])
    C₂ = centroid_of_area_enclosed_by_quadratic_segment(t[2], t[3], t[5])
    A₃ = area_enclosed_by_quadratic_segment(t[3], t[1], t[6])
    C₃ = centroid_of_area_enclosed_by_quadratic_segment(t[3], t[1], t[6])
    return (Aₜ*Cₜ + A₁ * C₁ + A₂ * C₂ + A₃ * C₃) / (Aₜ + A₁ + A₂ + A₃)
end

# -- Edges --

function edge(i::Integer, t::QuadraticTriangle)
    # Assumes 1 ≤ i ≤ 3.
    if i < 3
        return QuadraticSegment(t[i], t[i+1], t[i+3])
    else
        return QuadraticSegment(t[3], t[1], t[6])
    end
end

edges(t::QuadraticTriangle) = (edge(i, t) for i in 1:3)

# -- Bounding box --

function bounding_box(q::QuadraticTriangle)
    return bounding_box(edge(1, q)) ∪
           bounding_box(edge(2, q)) ∪
           bounding_box(edge(3, q))
end

# -- In --    
      
Base.in(P::Point{2}, t::QuadraticTriangle{2}) = all(edge -> isleft(P, edge), edges(t))

# -- Triangulation --

# N is the number of segments to divide each edge into.
# Return a Vector of the 2 * N^2 triangles that approximately partition 
# the quadratic quadrilateral.
function triangulate(t::QuadraticTriangle{D, T}, N::Integer) where {D, T}
    # Walk up the triangle in parametric coordinates (r as fast variable,
    # s as slow variable).
    # r is incremented along each row, forming two triangles (1,2,3) and
    # (3,2,4), at each step, until the last r value, which is a single triangle.
    # 3 --- 4        3 
    # | \   |        | \   
    # |   \ |        |   \  
    # 1 --- 2 ...... 1 --- 2 
    triangles = Vector{Triangle{D, T}}(undef, N^2)
    Δ = T(1) / N # Δ for r and s
    s1 = T(0)
    ntri = 0
    for s = 0:(N - 1)
        r1 = T(0)
        s2 = s1 + Δ
        p1 = t(r1, s1)
        p3 = t(r1, s2)
        for r = 0:(N - 2 - s)
            r2 = r1 + Δ
            p2 = t(r2, s1)
            p4 = t(r2, s2)
            triangles[ntri + 2 * r + 1] = Triangle(p1, p2, p3)
            triangles[ntri + 2 * r + 2] = Triangle(p3, p2, p4)
            r1 = r2
            p1 = p2
            p3 = p4
        end
        ntri += 2*(N - s) - 1
        triangles[ntri] = Triangle(p1, t(r1 + Δ, s1), p3)
        s1 = s2
    end
    return triangles
end

# -- IO --

function Base.show(io::IO, t::QuadraticTriangle{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "QuadraticTriangle", D, type_char, '(', 
        t.vertices[1], ", ", 
        t.vertices[2], ", ", 
        t.vertices[3], ", ",
        t.vertices[4], ", ",
        t.vertices[5], ", ",
        t.vertices[6], ')')
end
