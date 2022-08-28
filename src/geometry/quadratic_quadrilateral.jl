export QuadraticQuadrilateral,
       QuadraticQuadrilateral2,
       QuadraticQuadrilateral2f,
       QuadraticQuadrilateral2d

export interpolate_quadratic_quadrilateral,
       jacobian,
       quadratic_quadrilateral_jacobian,
       area,
       centroid,
       edge,
       edges,
       bounding_box,
       triangulate

# QUADRATIC QUADRILATERAL
# -----------------------------------------------------------------------------
#
# A quadratic quadrilateral represented by 8 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct QuadraticQuadrilateral{D, T}
    vertices::Vec{8, Point{D, T}}
end

# -- Type aliases --

const QuadraticQuadrilateral2  = QuadraticQuadrilateral{2}
const QuadraticQuadrilateral2f = QuadraticQuadrilateral2{Float32}
const QuadraticQuadrilateral2d = QuadraticQuadrilateral2{Float64}

# -- Base --

Base.getindex(Q::QuadraticQuadrilateral, i) = Q.vertices[i]
Base.broadcastable(Q::QuadraticQuadrilateral) = Ref(Q)

# -- Constructors --

function QuadraticQuadrilateral(
        P1::Point{D, T},
        P2::Point{D, T},
        P3::Point{D, T},
        P4::Point{D, T},
        P5::Point{D, T},
        P6::Point{D, T},
        P7::Point{D, T},
        P8::Point{D, T}) where {D, T}
    return QuadraticQuadrilateral{D, T}(Vec(P1, P2, P3, P4, P5, P6, P7, P8))
end

# -- Interpolation --

# Assumes the base linear quadrilateral is convex
function interpolate_quadratic_quadrilateral(P1::T, P2::T, P3::T, P4::T,
                                             P5::T, P6::T, P7::T, P8::T, r, s) where {T}
    ξ = 2 * r - 1
    η = 2 * s - 1
    return ((1 - ξ) * (1 - η) * (-ξ - η - 1) / 4) * P1 +
           ((1 + ξ) * (1 - η) * ( ξ - η - 1) / 4) * P2 +
           ((1 + ξ) * (1 + η) * ( ξ + η - 1) / 4) * P3 +
           ((1 - ξ) * (1 + η) * (-ξ + η - 1) / 4) * P4 +
                      ((1 - ξ * ξ) * (1 - η) / 2) * P5 +
                      ((1 - η * η) * (1 + ξ) / 2) * P6 +
                      ((1 - ξ * ξ) * (1 + η) / 2) * P7 +
                      ((1 - η * η) * (1 - ξ) / 2) * P8
end

function interpolate_quadratic_quadrilateral(vertices::Vec{8}, r, s)
    ξ = 2 * r - 1
    η = 2 * s - 1
    return ((1 - ξ) * (1 - η) * (-ξ - η - 1) / 4) * vertices[1] +
           ((1 + ξ) * (1 - η) * ( ξ - η - 1) / 4) * vertices[2] +
           ((1 + ξ) * (1 + η) * ( ξ + η - 1) / 4) * vertices[3] +
           ((1 - ξ) * (1 + η) * (-ξ + η - 1) / 4) * vertices[4] +
                      ((1 - ξ * ξ) * (1 - η) / 2) * vertices[5] +
                      ((1 - η * η) * (1 + ξ) / 2) * vertices[6] +
                      ((1 - ξ * ξ) * (1 + η) / 2) * vertices[7] +
                      ((1 - η * η) * (1 - ξ) / 2) * vertices[8]
end

function (Q::QuadraticQuadrilateral{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadratic_quadrilateral(Q.vertices, r, s)
end

# -- Jacobian --

# Assumes the base linear quadrilateral is convex
function quadratic_quadrilateral_jacobian(P1::T, P2::T, P3::T, P4::T,
                                          P5::T, P6::T, P7::T, P8::T, r, s) where {T}
    ξ = 2 * r - 1
    η = 2 * s - 1
    ∂r = (η * (1 - η) / 2) * (P1 - P2) +
         (η * (1 + η) / 2) * (P3 - P4) +
         (ξ * (1 - η)    ) * (
                             (P1 - P5) +
                             (P2 - P5)
                             ) +
         (ξ * (1 + η)    ) * (
                             (P3 - P7) +
                             (P4 - P7)
                             ) +
         ((1 + η)*(1 - η)) * (P6 - P8)

    ∂s = (ξ * (1 - ξ) / 2) * (P1 - P4) +
         (ξ * (1 + ξ) / 2) * (P3 - P2) +
         (η * (1 - ξ)    ) * (
                             (P1 - P8) +
                             (P4 - P8)
                             ) +
         (η * (1 + ξ)    ) * (
                             (P3 - P6) +
                             (P2 - P6)
                             ) +
         ((1 + ξ)*(1 - ξ)) * (P7 - P5)

    return Mat(∂r, ∂s)
end

function quadratic_quadrilateral_jacobian(vertices::Vec{8}, r, s)
    ξ = 2 * r - 1
    η = 2 * s - 1
    ∂r = (η * (1 - η) / 2) * (vertices[1] - vertices[2]) +
         (η * (1 + η) / 2) * (vertices[3] - vertices[4]) +
         (ξ * (1 - η)    ) * (
                             (vertices[1] - vertices[5]) +
                             (vertices[2] - vertices[5])
                             ) +
         (ξ * (1 + η)    ) * (
                             (vertices[3] - vertices[7]) +
                             (vertices[4] - vertices[7])
                             ) +
         ((1 + η)*(1 - η)) * (vertices[6] - vertices[8])

    ∂s = (ξ * (1 - ξ) / 2) * (vertices[1] - vertices[4]) +
         (ξ * (1 + ξ) / 2) * (vertices[3] - vertices[2]) +
         (η * (1 - ξ)    ) * (
                             (vertices[1] - vertices[8]) +
                             (vertices[4] - vertices[8])
                             ) +
         (η * (1 + ξ)    ) * (
                             (vertices[3] - vertices[6]) +
                             (vertices[2] - vertices[6])
                             ) +
         ((1 + ξ)*(1 - ξ)) * (vertices[7] - vertices[5])

    return Mat(∂r, ∂s)
end

function jacobian(Q::QuadraticQuadrilateral{D, T}, r::T, s::T) where {D, T}
    return quadratic_quadrilateral_jacobian(Q.vertices, r, s)
end

# -- Measure --

# Assumes the base linear quadrilateral is convex
function area(Q::QuadraticQuadrilateral{2, T}) where {T}
    # The area enclosed by the 4 quadratic edges + the area enclosed
    # by the quadrilateral (P1, P2, P3, P4)
    edge_area = T(2//3) * ((Q[5] - Q[1]) × (Q[2] - Q[1])  +
                           (Q[6] - Q[2]) × (Q[3] - Q[2])  +
                           (Q[7] - Q[3]) × (Q[4] - Q[3])  +
                           (Q[8] - Q[4]) × (Q[1] - Q[4]))

    quad_area = ((Q[2] - Q[1]) × (Q[3] - Q[1]) -
                 (Q[4] - Q[1]) × (Q[3] - Q[1]))  / 2

    return edge_area + quad_area
end

# -- Centroid --

# Assumes the base linear quadrilateral is convex
function centroid(Q::QuadraticQuadrilateral{2, T}) where {T}
    # By geometric decomposition into a quadrilateral and the 4 areas
    # enclosed by the quadratic edges.
    aq = quadrilateral_area(Q[1], Q[2], Q[3], Q[4])
    Cq = quadrilateral_centroid(Q[1], Q[2], Q[3], Q[4])
    a₁ = area_enclosed_by_quadratic_segment(Q[1], Q[2], Q[5])
    a₂ = area_enclosed_by_quadratic_segment(Q[2], Q[3], Q[6])
    a₃ = area_enclosed_by_quadratic_segment(Q[3], Q[4], Q[7])
    a₄ = area_enclosed_by_quadratic_segment(Q[4], Q[1], Q[8])
    C₁ = centroid_of_area_enclosed_by_quadratic_segment(Q[1], Q[2], Q[5])
    C₂ = centroid_of_area_enclosed_by_quadratic_segment(Q[2], Q[3], Q[6])
    C₃ = centroid_of_area_enclosed_by_quadratic_segment(Q[3], Q[4], Q[7])
    C₄ = centroid_of_area_enclosed_by_quadratic_segment(Q[4], Q[1], Q[8])
    return (aq * Cq + a₁ * C₁ + a₂ * C₂ + a₃ * C₃ + a₄ * C₄) / (aq + a₁ + a₂ + a₃ + a₄)
end

# -- Edges --

function edge(i::Integer, Q::QuadraticQuadrilateral)
    # Assumes 1 ≤ i ≤ 4.
    if i < 4
        return QuadraticSegment(Q[i], Q[i+1], Q[i+4])
    else
        return QuadraticSegment(Q[4], Q[1], Q[8])
    end
end

edges(Q::QuadraticQuadrilateral) = (edge(i, Q) for i in 1:4)

# -- Bounding box --

function bounding_box(Q::QuadraticQuadrilateral)
    return (bounding_box(edge(1, Q)) ∪ bounding_box(edge(2, Q))) ∪
           (bounding_box(edge(3, Q)) ∪ bounding_box(edge(4, Q)))
end

# -- In --

function Base.in(P::Point{2}, Q::QuadraticQuadrilateral{2})
    return all(edge -> isleft(P, edge), edges(Q))
end

# -- Triangulation --

# N is the number of segments to divide each edge into.
# Return a Vector of the 2 * N^2 triangles that approximately partition
# the quadratic quadrilateral.
function triangulate(Q::QuadraticQuadrilateral{D, T}, N::Integer) where {D, T}
    # Walk up the quadrilateral in parametric coordinates (r as fast variable,
    # s as slow variable).
    # r is incremented along each row, forming two triangles (1,2,3) and
    # (3,2,4), at each step.
    # 3 --- 4
    # | \   |
    # |   \ |
    # 1 --- 2
    triangles = Vector{Triangle{D, T}}(undef, 2 * N^2)
    Δ = T(1) / N # Δ for r and s
    s1 = T(0)
    for s = 0:(N - 1)
        r1 = T(0)
        s2 = s1 + Δ
        P1 = Q(r1, s1)
        P3 = Q(r1, s2)
        for r = 0:(N - 1)
            r2 = r1 + Δ
            P2 = Q(r2, s1)
            P4 = Q(r2, s2)
            triangles[2 * (N * s + r) + 1] = Triangle(P1, P2, P3)
            triangles[2 * (N * s + r) + 2] = Triangle(P3, P2, P4)
            r1 = r2
            P1 = P2
            P3 = P4
        end
        s1 = s2
    end
    return triangles
end

# -- IO --

function Base.show(io::IO, Q::QuadraticQuadrilateral{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "Quadrilateral", D, type_char, '(',
        Q.vertices[1], ", ",
        Q.vertices[2], ", ",
        Q.vertices[3], ", ",
        Q.vertices[4], ", ",
        Q.vertices[5], ", ",
        Q.vertices[6], ", ",
        Q.vertices[7], ", ",
        Q.vertices[8], ')')
end
