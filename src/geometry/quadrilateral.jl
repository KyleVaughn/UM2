export Quadrilateral,
       Quadrilateral2,
       Quadrilateral2f,
       Quadrilateral2d

export interpolate_quadrilateral,
       jacobian,
       quadrilateral_jacobian,
       area,
       quadrilateral_area,
       centroid,
       quadrilateral_centroid,
       edge,
       edges,
       bounding_box,
       triangulate

# QUADRILATERAL
# -----------------------------------------------------------------------------
#
# A quadrilateral represented by its 4 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct Quadrilateral{D, T} <: Polygon{D, T}
    vertices::Vec{4, Point{D, T}}
end

# -- Type aliases --

const Quadrilateral2  = Quadrilateral{2}
const Quadrilateral2f = Quadrilateral2{Float32}
const Quadrilateral2d = Quadrilateral2{Float64}

# -- Base --

Base.getindex(t::Quadrilateral, i) = t.vertices[i]
Base.broadcastable(t::Quadrilateral) = Ref(t)

# -- Constructors --

function Quadrilateral(
        p1::Point{D, T},
        p2::Point{D, T},
        p3::Point{D, T},
        p4::Point{D, T}) where {D, T}
    return Quadrilateral{D, T}(Vec(p1, p2, p3, p4))
end

# -- Interpolation --

# Assumes a convex quadrilateral
function interpolate_quadrilateral(p1::T, p2::T, p3::T, p4::T, r, s) where {T}
    return ((1 - r) * (1 - s)) * p1 +
           (     r  * (1 - s)) * p2 +
           (     r  *      s ) * p3 +
           ((1 - r) *      s ) * p4
end

function interpolate_quadrilateral(vertices::Vec{4}, r, s)
    return ((1 - r) * (1 - s)) * vertices[1] +
           (     r  * (1 - s)) * vertices[2] +
           (     r  *      s ) * vertices[3] +
           ((1 - r) *      s ) * vertices[4]
end

function (q::Quadrilateral{D, T})(r::T) where {D, T}
    return interpolate_quadrilateral(q.vertices, r, s)
end

# -- Jacobian --

# Assumes a convex quadrilateral
function quadrilateral_jacobian(p1::T, p2::T, p3::T, p4::T, r, s) where {T}
    ∂r = (1 - s) * (p2 - p1) - s * (p4 - p3)
    ∂s = (1 - r) * (p4 - p1) - r * (p2 - p3)
    return Mat(∂r, ∂s)
end

function quadrilateral_jacobian(vertices::Vec{4}, r, s)
    ∂r = (1 - s) * (vertices[2] - vertices[1]) - s * (vertices[4] - vertices[3])
    ∂s = (1 - r) * (vertices[4] - vertices[1]) - r * (vertices[2] - vertices[3])
    return Mat(∂r, ∂s)
end

function jacobian(q::Quadrilateral{D, T}, r::T, s::T) where {D, T}
    return quadrilateral_jacobian(q.vertices, r, s)
end

# -- Measure --

# Assumes a convex quadrilateral
function area(q::Quadrilateral{2})
    return ((q[3] - q[1]) × (q[4] - q[2])) / 2
end

function quadrilateral_area(p1::P, p2::P, p3::P, p4::P) where {P <: Point{2}}
    return ((p3 - p1) × (p4 - p2)) / 2
end

# -- Centroid --

# Assumes a convex quadrilateral
function centroid(q::Quadrilateral{2})
    # By geometric decomposition into two triangles
    v₁₂ = q[2] - q[1]
    v₁₃ = q[3] - q[1]
    v₁₄ = q[4] - q[1]
    A₁ = v₁₂ × v₁₃
    A₂ = v₁₃ × v₁₄
    P₁₃ = q[1] + q[3]
    return (A₁ * (P₁₃ + q[2]) + A₂ * (P₁₃ + q[4])) / (3 * (A₁ + A₂))
end

function quadrilateral_centroid(p1::P, p2::P, p3::P, p4::P) where {P <: Point{2}}
    v₁₂ = p2 - p1
    v₁₃ = p3 - p1
    v₁₄ = p4 - p1
    A₁ = v₁₂ × v₁₃
    A₂ = v₁₃ × v₁₄
    P₁₃ = p1 + p3
    return (A₁ * (P₁₃ + p2) + A₂ * (P₁₃ + p4)) / (3 * (A₁ + A₂))
end

# -- Edges --

function edge(i::Integer, q::Quadrilateral)
    # Assumes 1 ≤ i ≤ 4.
    if i < 4
        return LineSegment(t[i], t[i+1])
    else
        return LineSegment(t[4], t[1])
    end
end

edges(q::Quadrilateral) = (edge(i, q) for i in 1:4)

# -- Bounding box --

function bounding_box(q::Quadrilateral)
    return bounding_box(q.vertices)
end

# -- In --    
      
Base.in(P::Point{2}, q::Quadrilateral{2}) = all(edge -> isleft(P, edge), edges(q))

# -- Triangulation --

# Assumes a convex quadrilateral
function triangulate(q::Quadrilateral{2})
    return Vec(
        Triangle(q[1], q[2], q[3]),
        Triangle(q[1], q[3], q[4])
       )
end

# -- IO --

function Base.show(io::IO, q::Quadrilateral{D, T}) where {D, T}
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
        q.vertices[4], ')')
end
