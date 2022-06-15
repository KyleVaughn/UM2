function Base.intersect(l::LineSegment{Point{2, T}},
                        conn::Vector{<:LineSegment},
                        points::Vector{Point{2, T}}) where {T}
    T_INF_POINT = T(INF_POINT)
    p0 = l[1]
    intersections = Point{2, T}[]
    nintersections = 0
    for line in conn
        p = l ∩ materialize(line, points)
        if p[1] !== T_INF_POINT
            index = getsortedfirst(p0, intersections, p)
            if nintersections < index || intersections[index] !== p
                insert!(intersections, index, p)
                nintersections += 1
            end
        end
    end
    return intersections
end

function Base.intersect(l::LineSegment{Point{2, T}},
                        conn::Vector{<:Polygon},
                        points::Vector{Point{2, T}}) where {T}
    p0 = l[1]
    intersections = Point{2, T}[]
    for poly in conn
        _insert_valid_intersections!(intersections, p0, l ∩ materialize(poly, points))
    end
    return intersections
end
