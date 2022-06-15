function _insert_valid_intersections!(intersections::Vector{Point{D, T}},
                                      p0::Point{D, T},
                                      points::Vec{N, Point{D, T}},
                                      ) where {N, D, T}
    T_INF_POINT = T(INF_POINT)
    nintersections = length(intersections)
    for p in points
        if p[1] !== T_INF_POINT
            index = getsortedfirst(p0, intersections, p)
            if nintersections < index || intersections[index] !== p
                insert!(intersections, index, p)
                nintersections += 1
            end
        end
    end
    return nothing
end
