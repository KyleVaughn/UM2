export edge,
       edge_iterator,
       bounding_box

# POLYGON 
# -----------------------------------------------------------------------------
#
# A polygon represented by its vertices.
# These N vertices are D-dimensional points of type T.
#

# -- Edges --

function polygon_ev_conn(i::Integer, fv_conn::NTuple{N, I}) where {N, I <: Integer}
    # Assumes 1 ≤ i ≤ N.
    if i < N
        return (fv_conn[i], fv_conn[i + 1])
    else
        return (fv_conn[N], fv_conn[1])
    end
end

function polygon_ev_conn_iterator(fv_conn::NTuple{N, I}) where {N, I <: Integer}
    return (polygon_ev_conn(i, fv_conn) for i in 1:N)
end

function edge(i::Integer, P::Polygon{N}) where {N}
    # Assumes 1 ≤ i ≤ N.
    if i < N
        return LineSegment(P[i], P[i + 1])
    else
        return LineSegment(P[N], P[1])
    end
end

edge_iterator(P::Polygon{N}) where {N} = (edge(i, P) for i in 1:N)

# -- Bounding box --

bounding_box(P::Polygon) = bounding_box(P.vertices)
