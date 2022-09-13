export QuadraticPolygon

export vertices,
       edge,
       edge_iterator

# QUADRATIC POLYGON 
# -----------------------------------------------------------------------------
#
# A quadratic polygon represented by the connectivity of a set of edge points.
# These N points are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct QuadraticPolygon{N, D, T}
    vertices::NTuple{N, Point{D, T}}
end

# -- Type aliases --

const QPolygon = QuadraticPolygon

# -- Constructors --

function QPolygon{N}(vertices::NTuple{N, Point{D, T}}) where {N, D, T} 
    return QPolygon{N, D, T}(vertices)
end

# -- Base --

Base.getindex(QP::QPolygon, i::Integer) = QP.vertices[i]

# -- Accessors --

vertices(QP::QPolygon) = QP.vertices

# -- Edges --

function qpolygon_ev_conn(i::Integer, fv_conn::NTuple{N, I}) where {N, I <: Integer}
    # Assumes 1 ≤ i ≤ M.
    M = N ÷ 2
    if i < M
        return (fv_conn[i], fv_conn[i + 1], fv_conn[i + M])
    else
        return (fv_conn[M], fv_conn[1], fv_conn[N])
    end
end

function qpolygon_ev_conn_iterator(fv_conn::NTuple{N, I}) where {N, I <: Integer}
    M = N ÷ 2
    return (qpolygon_ev_conn(i, fv_conn) for i in 1:M)
end

function edge(i::Integer, QP::QPolygon{N}) where {N}
    # Assumes 1 ≤ i ≤ M.
    M = N ÷ 2
    if i < M
        return QuadraticSegment(QP[i], QP[i + 1], QP[i + M])
    else
        return QuadraticSegment(QP[M], QP[1], QP[N])
    end
end

function edge_iterator(QP::QPolygon{N}) where {N}
    M = N ÷ 2
    return (edge(i, QP) for i in 1:M)
end

# -- In --    
      
Base.in(P::Point2, QP::QPolygon) = all(edge -> isleft(P, edge), edge_iterator(QP))
