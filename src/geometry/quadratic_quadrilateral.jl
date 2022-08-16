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

struct QuadraticQuadrilateral{D, T} <: QuadraticPolygon{D, T}
    vertices::Vec{8, Point{D, T}}
end

# -- Type aliases --

const QuadraticQuadrilateral2  = QuadraticQuadrilateral{2}
const QuadraticQuadrilateral2f = QuadraticQuadrilateral2{Float32}
const QuadraticQuadrilateral2d = QuadraticQuadrilateral2{Float64}

# -- Base --

Base.getindex(t::QuadraticQuadrilateral, i) = t.vertices[i]
Base.broadcastable(t::QuadraticQuadrilateral) = Ref(t)

# -- Constructors --

function QuadraticQuadrilateral(
        p1::Point{D, T},
        p2::Point{D, T},
        p3::Point{D, T},
        p4::Point{D, T},
        p5::Point{D, T},
        p6::Point{D, T},
        p7::Point{D, T},
        p8::Point{D, T}) where {D, T}
    return QuadraticQuadrilateral{D, T}(Vec(p1, p2, p3, p4, p5, p6, p7, p8))
end

# -- Interpolation --

# Assumes the base linear quadrilateral is convex
function interpolate_quadratic_quadrilateral(p1::T, p2::T, p3::T, p4::T,
                                             p5::T, p6::T, p7::T, p8::T, r, s) where {T}
    ξ = 2 * r - 1
    η = 2 * s - 1
    return ((1 - ξ) * (1 - η) * (-ξ - η - 1) / 4) * p1 +
           ((1 + ξ) * (1 - η) * ( ξ - η - 1) / 4) * p2 +
           ((1 + ξ) * (1 + η) * ( ξ + η - 1) / 4) * p3 +
           ((1 - ξ) * (1 + η) * (-ξ + η - 1) / 4) * p4 +
                      ((1 - ξ * ξ) * (1 - η) / 2) * p5 +
                      ((1 - η * η) * (1 + ξ) / 2) * p6 +
                      ((1 - ξ * ξ) * (1 + η) / 2) * p7 +
                      ((1 - η * η) * (1 - ξ) / 2) * p8
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

function (q::QuadraticQuadrilateral{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadratic_quadrilateral(q.vertices, r, s)
end

# -- Jacobian --

# Assumes the base linear quadrilateral is convex
function quadratic_quadrilateral_jacobian(p1::T, p2::T, p3::T, p4::T,
                                          p5::T, p6::T, p7::T, p8::T, r, s) where {T}
    ξ = 2 * r - 1
    η = 2 * s - 1
    ∂r = (η * (1 - η) / 2) * (p1 - p2) +
         (η * (1 + η) / 2) * (p3 - p4) +
         (ξ * (1 - η)    ) * (
                             (p1 - p5) +
                             (p2 - p5)
                             ) +
         (ξ * (1 + η)    ) * (
                             (p3 - p7) +
                             (p4 - p7)
                             ) +
         ((1 + η)*(1 - η)) * (p6 - p8)

    ∂s = (ξ * (1 - ξ) / 2) * (p1 - p4) +
         (ξ * (1 + ξ) / 2) * (p3 - p2) +
         (η * (1 - ξ)    ) * (
                             (p1 - p8) +
                             (p4 - p8)
                             ) +
         (η * (1 + ξ)    ) * (
                             (p3 - p6) +
                             (p2 - p6)
                             ) +
         ((1 + ξ)*(1 - ξ)) * (p7 - p5)

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

function jacobian(q::QuadraticQuadrilateral{D, T}, r::T, s::T) where {D, T}
    return quadratic_quadrilateral_jacobian(q.vertices, r, s)
end

# -- Measure --

# Assumes the base linear quadrilateral is convex
function area(q::QuadraticQuadrilateral{2, T}) where {T}
    # The area enclosed by the 4 quadratic edges + the area enclosed
    # by the quadrilateral (p1, p2, p3, p4)
    edge_area = T(2//3) * ((q[5] - q[1]) × (q[2] - q[1])  +
                           (q[6] - q[2]) × (q[3] - q[2])  +
                           (q[7] - q[3]) × (q[4] - q[3])  +
                           (q[8] - q[4]) × (q[1] - q[4]))

    quad_area = ((q[2] - q[1]) × (q[3] - q[1]) -
                 (q[4] - q[1]) × (q[3] - q[1]))  / 2

    return edge_area + quad_area
end

# -- Centroid --

# Assumes the base linear quadrilateral is convex
function centroid(q::QuadraticQuadrilateral{2, T}) where {T}
    # By geometric decomposition into a quadrilateral and the 4 areas
    # enclosed by the quadratic edges.
    Aq = quadrilateral_area(q[1], q[2], q[3], q[4])
    Cq = quadrilateral_centroid(q[1], q[2], q[3], q[4])
    A₁ = area_enclosed_by_quadratic_segment(q[1], q[2], q[5])
    A₂ = area_enclosed_by_quadratic_segment(q[2], q[3], q[6])
    A₃ = area_enclosed_by_quadratic_segment(q[3], q[4], q[7])
    A₄ = area_enclosed_by_quadratic_segment(q[4], q[1], q[8])
    C₁ = centroid_of_area_enclosed_by_quadratic_segment(q[1], q[2], q[5])
    C₂ = centroid_of_area_enclosed_by_quadratic_segment(q[2], q[3], q[6])
    C₃ = centroid_of_area_enclosed_by_quadratic_segment(q[3], q[4], q[7])
    C₄ = centroid_of_area_enclosed_by_quadratic_segment(q[4], q[1], q[8])
    return (Aq * Cq + A₁ * C₁ + A₂ * C₂ + A₃ * C₃ + A₄ * C₄) / (Aq + A₁ + A₂ + A₃ + A₄)
end

# -- Edges --

function edge(i::Integer, q::QuadraticQuadrilateral)
    # Assumes 1 ≤ i ≤ 4.
    if i < 4
        return QuadraticSegment(q[i], q[i+1], q[i+4])
    else
        return QuadraticSegment(q[4], q[1], q[8])
    end
end

edges(q::QuadraticQuadrilateral) = (edge(i, q) for i in 1:4)

# -- Bounding box --

function bounding_box(q::QuadraticQuadrilateral)
    return bounding_box(edge(1, q)) ∪
           bounding_box(edge(2, q)) ∪
           bounding_box(edge(3, q)) ∪
           bounding_box(edge(4, q))
end

# -- In --

function Base.in(P::Point{2}, q::QuadraticQuadrilateral{2})
    return all(edge -> isleft(P, edge), edges(q))
end

# -- Triangulation --

# N is the number of segments to divide each edge into.
# Return a Vector of the N^2 triangles that approximately partition
# the quadratic triangle.
function triangulate(q::QuadraticQuadrilateral{D, T}, N::Integer) where {D, T}
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
        p1 = q(r1, s1)
        p3 = q(r1, s2)
        for r = 0:(N - 1)
            r2 = r1 + Δ
            p2 = q(r2, s1)
            p4 = q(r2, s2)
            triangles[2 * (N * s + r) + 1] = Triangle(p1, p2, p3)
            triangles[2 * (N * s + r) + 2] = Triangle(p3, p2, p4)
            r1 = r2
            p1 = p2
            p3 = p4
        end
        s1 = s2
    end
    return triangles
end

# -- IO --

function Base.show(io::IO, q::QuadraticQuadrilateral{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "Quadrilateral", D, type_char, '(',
        q.vertices[1], ", ",
        q.vertices[2], ", ",
        q.vertices[3], ", ",
        q.vertices[4], ", ",
        q.vertices[5], ", ",
        q.vertices[6], ", ",
        q.vertices[7], ", ",
        q.vertices[8], ')')
end
