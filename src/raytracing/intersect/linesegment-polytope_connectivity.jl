function Base.intersect(l::LineSegment{Point{2, T}},
                        conn::Vector{<:LineSegment},
                        points::Vector{Point{2, T}}) where {T}
    T_INF_POINT = T(INF_POINT)
    P₀ = l[1]
    intersections = Point{2, T}[]
    nintersections = 0
    for line in conn
        P = l ∩ materialize(line, points)
        if P[1] !== T_INF_POINT
            index = getsortedfirst(P₀, intersections, P)
            if nintersections < index || intersections[index] !== P
                insert!(intersections, index, P)
                nintersections += 1
            end
        end
    end
    return intersections
end

function Base.intersect(l::LineSegment{Point{2, T}},
                        conn::Vector{<:Polygon},
                        points::Vector{Point{2, T}}) where {T}
    P₀ = l[1]
    intersections = Point{2, T}[]
    for poly in conn
        _insert_valid_intersections!(intersections, P₀, l ∩ materialize(poly, points))
    end
    return intersections
end
