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
       edge_iterator,
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

struct Quadrilateral{D, T}
    vertices::Vec{4, Point{D, T}}
end

# -- Type aliases --

const Quadrilateral2  = Quadrilateral{2}
const Quadrilateral2f = Quadrilateral2{Float32}
const Quadrilateral2d = Quadrilateral2{Float64}

# -- Base --

Base.getindex(Q::Quadrilateral, i) = Q.vertices[i]
Base.broadcastable(Q::Quadrilateral) = Ref(Q)

# -- Constructors --

function Quadrilateral(
        P1::Point{D, T},
        P2::Point{D, T},
        P3::Point{D, T},
        P4::Point{D, T}) where {D, T}
    return Quadrilateral{D, T}(Vec(P1, P2, P3, P4))
end

# -- Interpolation --

# Assumes a convex quadrilateral
function interpolate_quadrilateral(P1::T, P2::T, P3::T, P4::T, r, s) where {T}
    return ((1 - r) * (1 - s)) * P1 +
           (     r  * (1 - s)) * P2 +
           (     r  *      s ) * P3 +
           ((1 - r) *      s ) * P4
end

function interpolate_quadrilateral(vertices::Vec{4}, r, s)
    return ((1 - r) * (1 - s)) * vertices[1] +
           (     r  * (1 - s)) * vertices[2] +
           (     r  *      s ) * vertices[3] +
           ((1 - r) *      s ) * vertices[4]
end

function (Q::Quadrilateral{D, T})(r::T) where {D, T}
    return interpolate_quadrilateral(Q.vertices, r, s)
end

# -- Jacobian --

# Assumes a convex quadrilateral
function quadrilateral_jacobian(P1::T, P2::T, P3::T, P4::T, r, s) where {T}
    ∂r = (1 - s) * (P2 - P1) - s * (P4 - P3)
    ∂s = (1 - r) * (P4 - P1) - r * (P2 - P3)
    return Mat(∂r, ∂s)
end

function quadrilateral_jacobian(vertices::Vec{4}, r, s)
    ∂r = (1 - s) * (vertices[2] - vertices[1]) - s * (vertices[4] - vertices[3])
    ∂s = (1 - r) * (vertices[4] - vertices[1]) - r * (vertices[2] - vertices[3])
    return Mat(∂r, ∂s)
end

function jacobian(Q::Quadrilateral{D, T}, r::T, s::T) where {D, T}
    return quadrilateral_jacobian(Q.vertices, r, s)
end

# -- Measure --

# Assumes a convex quadrilateral
function area(Q::Quadrilateral{2})
    return ((Q[3] - Q[1]) × (Q[4] - Q[2])) / 2
end

function quadrilateral_area(P1::P, P2::P, P3::P, P4::P) where {P <: Point{2}}
    return ((P3 - P1) × (P4 - P2)) / 2
end

# -- Centroid --

# Assumes a convex quadrilateral
function centroid(Q::Quadrilateral{2})
    # By geometric decomposition into two triangles
    v₁₂ = Q[2] - Q[1]
    v₁₃ = Q[3] - Q[1]
    v₁₄ = Q[4] - Q[1]
    a₁ = v₁₂ × v₁₃
    ma₂ = v₁₄ × v₁₃ # minus a₂. Flipped for potential SSE optimization
    P₁₃ = Q[1] + Q[3]
    return (a₁ * (P₁₃ + q[2]) - ma₂ * (P₁₃ + q[4])) / (3 * (a₁ - ma₂))
end

function quadrilateral_centroid(P1::P, P2::P, P3::P, P4::P) where {P <: Point{2}}
    v₁₂ = P2 - P1
    v₁₃ = P3 - P1
    v₁₄ = P4 - P1
    a₁ = v₁₂ × v₁₃
    ma₂ = v₁₄ × v₁₃ # minus a₂. Flipped for potential SSE optimization
    P₁₃ = P1 + P3
    return (a₁ * (P₁₃ + P2) - ma₂ * (P₁₃ + P4)) / (3 * (a₁ - ma₂))
end

# -- Edges --

function ev_conn(i::Integer, fv_conn::NTuple{4, I}) where {I <: Integer}    
    # Assumes 1 ≤ i ≤ 4.    
    if i < 4    
        return (fv_conn[i], fv_conn[i + 1])    
    else    
        return (fv_conn[4], fv_conn[1])    
    end    
end    
    
function ev_conn_iterator(fv_conn::NTuple{4, I}) where {I}    
    return (ev_conn(i, fv_conn) for i in 1:4)    
end    

function edge(i::Integer, Q::Quadrilateral)
    # Assumes 1 ≤ i ≤ 4.
    if i < 4
        return LineSegment(Q[i], Q[i+1])
    else
        return LineSegment(Q[4], Q[1])
    end
end
    
edge_iterator(Q::Quadrilateral) = (edge(i, Q) for i in 1:4)

# -- Bounding box --

function bounding_box(Q::Quadrilateral)
    return bounding_box(Q.vertices)
end

# -- In --    
      
Base.in(P::Point{2}, Q::Quadrilateral{2}) = all(edge -> isleft(P, edge), edge_iterator(Q))

# -- Triangulation --

# Assumes a convex quadrilateral
function triangulate(Q::Quadrilateral{2})
    return Vec(
        Triangle(Q[1], Q[2], Q[3]),
        Triangle(Q[1], Q[3], Q[4])
       )
end

# -- IO --

function Base.show(io::IO, Q::Quadrilateral{D, T}) where {D, T}
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
        Q.vertices[4], ')')
end
