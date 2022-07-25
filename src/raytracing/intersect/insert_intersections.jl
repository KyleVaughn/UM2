function _insert_valid_intersections!(intersections::Vector{Point{D, T}},
                                      p0::Point{D, T},
                                      points::Vec{N, Point{D, T}}) where {N, D, T}
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

function _insert_nonduplicate_intersections!(intersections::Vector{Point{D, T}},
                                            p0::Point{D, T},
                                            points::Vec{N, Point{D, T}}) where {N, D, T}
    T_INF_POINT = T(INF_POINT)
    nintersections = length(intersections)
    for p in points
        if p[1] !== T_INF_POINT
            index = getsortedfirst(p0, intersections, p)
            # index == 1
            #   The point will be inserted at the start.
            #   Need to ensure p ≉ intersections[1] 
            #   Equivalent to p ≉ intersections[index]

            # nintersections < index
            #   The point will be inserted at the end.
            #   Need to ensure p ≉ intersections[nintersections]
            #   Equivalent to p ≉ intersections[index - 1]
               
            # 1 < index ≤ nintersections
            #   The point will be inserted somewhere in the middle.
            #   Need to ensure p ≉ intersections[index - 1] && p ≉ intersections[index]
            if 1 < index ≤ nintersections
                if p ≉ intersections[index - 1] && p ≉ intersections[index]
                    insert!(intersections, index, p)
                    nintersections += 1
                end
            elseif index === 1
                if length(intersections) == 0 || p ≉ intersections[1]
                    insert!(intersections, index, p)
                    nintersections += 1
                end
            elseif nintersections < index     
                if p ≉ intersections[nintersections]
                    insert!(intersections, index, p)
                    nintersections += 1
                end
            end
        end
    end
    return nothing
end
