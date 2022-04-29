export Point, Point1, Point2, Point3, Point1f, Point2f, Point3f, Point1B,
       Point2B, Point3B
export coordinates, distance, distance², fast_isapprox, isCCW, midpoint, nan

"""
    Point{Dim, T}

A point in `Dim`-dimensional space with coordinates of type `T`.

The coordinates of the point are given with respect to the canonical
Euclidean basis.

## Examples

```julia
# 2D points
A = Point(0.0, 1.0) # double precision as expected
B = Point(0f0, 1f0) # single precision as expected
C = Point2(0, 1)    # explicitly ask for double precision
D = Point2f(0, 1)   # explicitly ask for single precision
E = Point2B(0, 1)   # explicitly ask for BigFloat

# 3D points
F = Point(1.0, 2.0, 3.0) # double precision as expected
G = Point(1f0, 2f0, 3f0) # single precision as expected
H = Point3(1, 2, 3)      # explicitly ask for double precision
I = Point3f(1, 2, 3)     # explicitly ask for single precision
J = Point3B(1, 2, 3)     # explicitly ask for BigFloat 
```

### Notes

- Type aliases are `Point1`, `Point2`, `Point3`, `Point1f`, `Point2f`, `Point3f`,
    `Point1B`, `Point2B`, `Point3B`.
"""
struct Point{Dim,T} <:AbstractVector{T}
    coords::SVector{Dim,T}
    Point{Dim,T}(coords::SVector{Dim,T}) where {Dim,T} = new{Dim,T}(coords)
end

# constructors
Point{Dim,T}(coords...) where {Dim,T} = Point{Dim,T}(SVector{Dim,T}(coords...))
Point(coords::SVector{Dim,T}) where {Dim,T} = Point{Dim,T}(coords)
Point(coords::AbstractVector{T}) where {T} = Point{length(coords),T}(coords)
Point(coords...) = Point(SVector(coords...))

# conversions
convert(::Type{Point{Dim,T}}, coords) where {Dim,T} = Point{Dim,T}(coords)
convert(::Type{Point{Dim,T}}, P::Point) where {Dim,T} = Point{Dim,T}(P.coords)
SVector(P::Point{Dim,T}) where {Dim,T} = P.coords

# type aliases
const Point1  = Point{1,Float64}
const Point2  = Point{2,Float64}
const Point3  = Point{3,Float64}
const Point1f = Point{1,Float32}
const Point2f = Point{2,Float32}
const Point3f = Point{3,Float32}
const Point1B = Point{1,BigFloat}
const Point2B = Point{2,BigFloat}
const Point3B = Point{3,BigFloat}

# abstract array interface
Base.size(P::Point) = Base.size(P.coords)
Base.getindex(P::Point, i::Int) = Base.getindex(P.coords, i) 
Base.IndexStyle(P::Point) = Base.IndexStyle(typeof(P.coords))

zero(::Type{Point{Dim,T}}) where {Dim,T} = Point{Dim,T}(@SVector zeros(T, Dim))
nan(::Type{Point{Dim,T}}) where {Dim,T} = Point{Dim,T}(@SVector fill(T(NaN), Dim))

"""
    coordinates(p::Point)

Return the coordinates of the point with respect to the
canonical Euclidean basis.
"""
coordinates(p::Point) = p.coords

"""
    -(A::Point, B::Point)

Return the [`Vec`](@ref) displacement from point `B` to point `A`.
"""
-(A::Point, B::Point) = A.coords - B.coords

"""
    +(A::Point, v::Vec)

Return the point at the end of the vector `v` placed
at a reference (or start) point `A`.
"""
+(A::Point, v::Vec) = Point(A.coords + v)

"""
    -(A::Point, v::Vec)

Return the point at the end of the vector `-v` placed
at a reference (or start) point `A`.
"""
-(A::Point, v::Vec) = Point(A.coords - v)

"""
    isapprox(A::Point, B::Point)

Return if two points are approximately equal (less than 1e-4 cm apart). 
"""
Base.isapprox(A::Point{Dim,T}, B::Point{Dim,T}) where {Dim,T} = norm²(A-B) < T(ϵ_Point^2)

"""
    distance(A::Point, B::Point)

Return the distance from point `A` to point `B`. 
"""
distance(A::Point, B::Point) = norm(B - A)

"""
    distance²(A::Point, B::Point)

Return the distance from point `A` to point `B` squared. 
"""
distance²(A::Point, B::Point) = norm²(B - A)

"""
    midpoint(A::Point, B::Point)

Return the midpoint of the line segment from point `A` to point `B`.
"""
midpoint(A::Point, B::Point) = Point((A.coords + B.coords)/2)

"""
    isCCW(A::Point{2}, B::Point{2}, C::Point{2})

If the triplet of 2-dimensional points is counter-clockwise oriented from A to B to C.
"""
isCCW(A::Point{2}, B::Point{2}, C::Point{2}) = 0 ≤ (B - A) × (C - A)

function Base.show(io::IO, point::Point)
    print(io, point.coords.data)
end
