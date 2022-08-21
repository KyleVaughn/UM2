function convert_arguments(LS::Type{<:LineSegments}, aab::AABox{2})
    p1 = Point(x_min(aab), y_min(aab))
    p2 = Point(x_max(aab), y_min(aab))
    p3 = Point(x_max(aab), y_max(aab))
    p4 = Point(x_min(aab), y_max(aab))
    return convert_arguments(LS, [ LineSegment(p1, p2),
                                   LineSegment(p2, p3),
                                   LineSegment(p3, p4),
                                   LineSegment(p4, p1) ])
end

function convert_arguments(LS::Type{<:LineSegments}, aabs::Vector{AABox{2, T}}) where {T}
    lines = Vector{LineSegment{D, T}}(undef, 4 * length(aabs))
    for i in eachindex(aabs)
        lines[4 * i - 3] = LineSegment(Point(x_min(aabs[i]), y_min(aabs[i])), 
                                       Point(x_max(aabs[i]), y_min(aabs[i])))
        lines[4 * i - 2] = LineSegment(Point(x_max(aabs[i]), y_min(aabs[i])),
                                       Point(x_max(aabs[i]), y_max(aabs[i])))
        lines[4 * i - 1] = LineSegment(Point(x_max(aabs[i]), y_max(aabs[i])),
                                       Point(x_min(aabs[i]), y_max(aabs[i])))
        lines[4 * i    ] = LineSegment(Point(x_min(aabs[i]), y_max(aabs[i])),
                                       Point(x_min(aabs[i]), y_min(aabs[i])))
    end
    return convert_arguments(LS, lines) 
end

function convert_arguments(M::Type{<:GLMakieMesh}, aab::AABox{2})
    p1 = Point(x_min(aab), y_min(aab))
    p2 = Point(x_max(aab), y_min(aab))
    p3 = Point(x_max(aab), y_max(aab))
    p4 = Point(x_min(aab), y_max(aab))
    verts = [ p1, p2, p3, p4 ]
    faces = [1 2 3;
             3 4 1]
    return convert_arguments(M, verts, faces)
end
