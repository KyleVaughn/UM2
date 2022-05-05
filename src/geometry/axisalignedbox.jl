export AABox
export measure, maxima, minima, xmin, ymin, zmin, xmax, ymax, zmax, Δx, Δy, Δz,
       facets, ridges, peaks, faces, edges, vertices

"""
    AABox(minima::Point{Dim,T}, maxima::Point{Dim,T})

A `Dim`-dimensional axis-aligned bounding box using two `Dim`-dimensional
points, representing the `minima` and `maxima` of the box. 
"""
struct AABox{Dim,T}
    minima::Point{Dim,T}
    maxima::Point{Dim,T}
    function AABox{Dim,T}(minima::Point{Dim,T}, maxima::Point{Dim,T}) where {Dim,T}
        if any(i->maxima[i] ≤ minima[i], 1:Dim)
            error("AABox maxima must be greater that the minima")
        end
        return new{Dim,T}(minima, maxima)
    end
end

# constructors
function AABox(minima::Point{Dim,T}, maxima::Point{Dim,T}) where {Dim,T}
    return AABox{Dim,T}(minima, maxima)
end
AABox(minima, maxima) = AABox(Point(minima), Point(maxima))

maxima(aab::AABox) = aab.maxima
minima(aab::AABox) = aab.minima

Base.isapprox(aab₁::AABox, aab₂::AABox) = minima(aab₁) ≈ minima(aab₂) && 
                                          maxima(aab₁) ≈ maxima(aab₂)
xmin(aab::AABox   ) = aab.minima[1] 
ymin(aab::AABox   ) = aab.minima[2] 
zmin(aab::AABox{3}) = aab.minima[3] 
xmax(aab::AABox   ) = aab.maxima[1] 
ymax(aab::AABox   ) = aab.maxima[2] 
zmax(aab::AABox{3}) = aab.maxima[3] 
Δx(aab::AABox   ) = xmax(aab) - xmin(aab) 
Δy(aab::AABox   ) = ymax(aab) - ymin(aab)
Δz(aab::AABox{3}) = zmax(aab) - zmin(aab)

measure(aab::AABox) = prod(maxima(aab) - minima(aab)) 

ridges(aab::AABox{2}) = vertices(aab)
facets(aab::AABox{2}) = edges(aab)

peaks( aab::AABox{3}) = vertices(aab)
ridges(aab::AABox{3}) = edges(aab)
facets(aab::AABox{3}) = faces(aab)

function vertices(aab::AABox{2})
    # Ordered CCW
    return Vec(Point(xmin(aab), ymin(aab)),
               Point(xmax(aab), ymin(aab)),
               Point(xmax(aab), ymax(aab)),
               Point(xmin(aab), ymax(aab))
              )
end
function edges(aab::AABox{2})
    v = vertices(aab)
    return Vec(LineSegment(v[1], v[2]),
               LineSegment(v[2], v[3]),
               LineSegment(v[3], v[4]),
               LineSegment(v[4], v[1])
              )
end

function vertices(aab::AABox{3})
    # in CCW order, low z then high z
    #      y
    #      ^
    #      |
    #      |
    #      |------> x
    #     /   
    #    /   
    #   𝘷
    #  z
    #
    #   4----3
    #  /    /|
    # 8----7 |
    # |    | 2
    # |    |/
    # 5----6
    return Vec(Point(xmin(aab), ymin(aab), zmin(aab)),
               Point(xmax(aab), ymin(aab), zmin(aab)),
               Point(xmax(aab), ymax(aab), zmin(aab)),
               Point(xmin(aab), ymax(aab), zmin(aab)),
               Point(xmin(aab), ymin(aab), zmax(aab)),             
               Point(xmax(aab), ymin(aab), zmax(aab)),
               Point(xmax(aab), ymax(aab), zmax(aab)),
               Point(xmin(aab), ymax(aab), zmax(aab))
              )
end

function edges(aab::AABox{3})
    # in CCW order, low z, then high z, then the segments that attach low and
    # high z in CCW order.
    #      y
    #      ^
    #      |
    #      |
    #      |------> x
    #     /   
    #    /   
    #   𝘷
    #  z
    #       3
    #    +----+
    # 11/ 7  /|
    #  +----+ | 2
    # 8|    |6+
    #  |    |/ 9
    #  +----+
    #     5
    v = vertices(aab)
    return Vec(LineSegment(v[1], v[2]), # lower z
               LineSegment(v[2], v[3]),
               LineSegment(v[3], v[4]),
               LineSegment(v[4], v[1]),
               LineSegment(v[5], v[6]), # upper z
               LineSegment(v[6], v[7]),
               LineSegment(v[7], v[8]),
               LineSegment(v[8], v[5]),
               LineSegment(v[1], v[5]), # lower, upper connections
               LineSegment(v[2], v[6]),
               LineSegment(v[3], v[7]),
               LineSegment(v[4], v[8])
              )
end

function faces(aab::AABox{3})
     v = vertices(aab)   
     return Vec(Quadrilateral(v[1], v[2], v[3], v[4]),
                Quadrilateral(v[5], v[6], v[7], v[8]),
                Quadrilateral(v[1], v[2], v[6], v[5]),
                Quadrilateral(v[2], v[3], v[7], v[6]),
                Quadrilateral(v[3], v[4], v[8], v[7]),
                Quadrilateral(v[4], v[1], v[5], v[8]),
               )
end

function Base.show(io::IO, aab::AABox)
    print(io, "AABox($(minima(aab)), $(maxima(aab)))")
end
