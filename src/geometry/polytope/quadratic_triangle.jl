export interpolate_quadratic_triangle,
       jacobian,
       quadratic_triangle_jacobian,
       area,
       centroid,
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

# -- Type aliases --

const QTriangle = QuadraticTriangle
const QTriangle2 = QuadraticTriangle2

# -- Constructors --

function QuadraticTriangle(
        P1::Point{D, T}, 
        P2::Point{D, T}, 
        P3::Point{D, T},
        P4::Point{D, T},
        P5::Point{D, T},
        P6::Point{D, T}) where {D, T}
    return QTriangle{D, T}((P1, P2, P3, P4, P5, P6))
end

# -- Interpolation --

function quadratic_triangle_weights(r, s)
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s), 
                      r           * ( 2 * r - 1),
                          s       * ( 2 * s - 1),
                  4 * r           * ( 1 - r - s),
                  4 * r           *           s ,
                      4 * s       * ( 1 - r - s))
end

function interpolate_quadratic_triangle(
        P1::T, P2::T, P3::T, P4::T, P5::T, P6::T, r, s) where {T}
    w = quadratic_triangle_weights(r, s)
    return w[1] * P1 + w[2] * P2 + w[3] * P3 + w[4] * P4 + w[5] * P5 + w[6] * P6
end

function interpolate_quadratic_triangle(vertices::NTuple{6}, r, s)
    return mapreduce(*, +, quadratic_triangle_weights(r, s), vertices)
end

function (QT::QTriangle{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadratic_triangle(QT.vertices, r, s)
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

function quadratic_triangle_jacobian(vertices::NTuple{6}, r, s)
    ∂r = (4 * r + 4 * s - 3) * (vertices[1] - vertices[4]) +    
         (4 * r         - 1) * (vertices[2] - vertices[4]) +    
         (        4 * s    ) * (vertices[5] - vertices[6])
    
    ∂s = (4 * r + 4 * s - 3) * (vertices[1] - vertices[6]) +
         (        4 * s - 1) * (vertices[3] - vertices[6]) +
         (4 * r            ) * (vertices[5] - vertices[4])
    
    return Mat(∂r, ∂s)
end

function jacobian(QT::QTriangle{D, T}, r::T, s::T) where {D, T}
    return quadratic_triangle_jacobian(QT.vertices, r, s)
end

# -- Measure --

function area(QT::QTriangle2{T}) where {T}
    # The area enclosed by the 3 quadratic edges + the area enclosed
    # by the triangle (P1, P2, P3)
    edge_area = T(2//3) * ((QT[4] - QT[1]) × (QT[2] - QT[1])  +
                           (QT[5] - QT[2]) × (QT[3] - QT[2])  +
                           (QT[6] - QT[3]) × (QT[1] - QT[3])) 
    tri_area = ((QT[2] - QT[1]) × (QT[3] - QT[1])) / 2 
    return edge_area + tri_area
end

# -- Centroid --

function centroid(QT::QTriangle2{T}) where {T}
    # By geometric decomposition into a triangle and the 3 areas
    # enclosed by the quadratic edges.
    aₜ = triangle_area(QT[1], QT[2], QT[3])
    Cₜ = triangle_centroid(QT[1], QT[2], QT[3])
    a₁ = area_enclosed_by_quadratic_segment(QT[1], QT[2], QT[4])
    a₂ = area_enclosed_by_quadratic_segment(QT[2], QT[3], QT[5])
    a₃ = area_enclosed_by_quadratic_segment(QT[3], QT[1], QT[6])
    C₁ = centroid_of_area_enclosed_by_quadratic_segment(QT[1], QT[2], QT[4])
    C₂ = centroid_of_area_enclosed_by_quadratic_segment(QT[2], QT[3], QT[5])
    C₃ = centroid_of_area_enclosed_by_quadratic_segment(QT[3], QT[1], QT[6])
    return (aₜ*Cₜ + a₁ * C₁ + a₂ * C₂ + a₃ * C₃) / (aₜ + a₁ + a₂ + a₃)
end

# -- Triangulation --

# N is the number of segments to divide each edge into.
# Return a Vector of the N^2 triangles that approximately partition 
# the quadratic triangle.
function triangulate(QT::QTriangle{D, T}, N::Integer) where {D, T}
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
        P1 = QT(r1, s1)
        P3 = QT(r1, s2)
        for r = 0:(N - 2 - s)
            r2 = r1 + Δ
            P2 = QT(r2, s1)
            P4 = QT(r2, s2)
            triangles[ntri + 2 * r + 1] = Triangle(P1, P2, P3)
            triangles[ntri + 2 * r + 2] = Triangle(P3, P2, P4)
            r1 = r2
            P1 = P2
            P3 = P4
        end
        ntri += 2*(N - s) - 1
        triangles[ntri] = Triangle(P1, QT(r1 + Δ, s1), P3)
        s1 = s2
    end
    return triangles
end

# -- IO --

function Base.show(io::IO, QT::QTriangle{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "QuadraticTriangle", D, type_char, '(', 
        QT.vertices[1], ", ", 
        QT.vertices[2], ", ", 
        QT.vertices[3], ", ",
        QT.vertices[4], ", ",
        QT.vertices[5], ", ",
        QT.vertices[6], ')')
end
