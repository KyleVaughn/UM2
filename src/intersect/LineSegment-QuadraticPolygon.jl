function Base.intersect(l::LineSegment2D{T}, poly::QuadraticPolygon{N, 2, T}
                       ) where {N, T <:Union{Float32, Float64}} 
    # Create the quadratic segments that make up the polygon and intersect each one
    points = zeros(MVector{N, Point2D{T}})
    npoints = 0x0000
    M = N ÷ 2
    for i ∈ 1:M-1
        hits, ipoints = l ∩ QuadraticSegment2D(poly[i], poly[i + 1], poly[i + M])
        for j in 1:hits
            npoints += 0x0001
            points[npoints] = ipoints[j]
        end
    end
    hits, ipoints = l ∩ QuadraticSegment2D(poly[M], poly[1], poly[N])
    for j in 1:hits
        npoints += 0x0001
        points[npoints] = ipoints[j]
    end
    return npoints, SVector(points)
end

# Cannot mutate BigFloats in an MVector, so we use a regular Vector
function Base.intersect(l::LineSegment2D{BigFloat}, poly::QuadraticPolygon{N, 2, BigFloat}
                       ) where {N} 
    # Create the quadratic segments that make up the polygon and intersect each one
    points = zeros(Point2D{BigFloat}, N)
    npoints = 0x0000
    M = N ÷ 2
    for i ∈ 1:M-1
        hits, ipoints = l ∩ QuadraticSegment2D(poly[i], poly[i + 1], poly[i + M])
        for j in 1:hits
            npoints += 0x0001
            points[npoints] = ipoints[j]
        end
    end
    hits, ipoints = l ∩ QuadraticSegment2D(poly[M], poly[1], poly[N])
    for j in 1:hits
        npoints += 0x0001
        points[npoints] = ipoints[j]
    end
    return npoints, SVector{N, Point2D{BigFloat}}(points)
end

