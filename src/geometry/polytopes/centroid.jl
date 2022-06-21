export centroid

# (https://en.wikipedia.org/wiki/Centroid#Of_a_polygon)
function centroid(poly::Polygon{N, Point{2, T}}) where {N, T}
    a = zero(T) # Scalar
    c = Vec{2, T}(0,0)
#    @inbounds for i in 1:N-1
    for i in Base.OneTo(N-1) 
        vec1 = coordinates(poly[i])
        vec2 = coordinates(poly[i + 1])
        subarea = vec1 × vec2 
        c += subarea*(vec1 + vec2)
        a += subarea
    end 
    vec1 = coordinates(poly[N])
    vec2 = coordinates(poly[1])
    subarea = vec1 × vec2
    c += subarea*(vec1 + vec2)
    a += subarea
    return Point(c/(3a))
end

centroid(tri::Triangle) = Point(mapreduce(coordinates, +, vertices(tri))/3)
