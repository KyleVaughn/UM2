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

function convert_arguments(LS::Type{<:LineSegments}, R::Vector{<:AABox})
    return convert_arguments(LS, vcat([convert_arguments(LS, aab)[1] for aab in R]...))
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
