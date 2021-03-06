export Point, Point1, Point2, Point3, Point1f, Point2f, Point3f, Point1B,
       Point2B, Point3B
export coordinates, distance, distance², isCCW, midpoint, nan, EPS_POINT,
       INF_POINT

"""
    Point{D, T}

A point in `D`-dimensional space with coordinates of type `T`.

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
struct Point{D, T} <: AbstractVector{T}
    coords::Vec{D, T}
    Point{D, T}(coords::Vec{D, T}) where {D, T} = new{D, T}(coords)
end

# constructors
Point{1, T}(x::X) where {X<:Number, T} = Point{1, T}(Vec{1, T}(x))
Point{2, T}(x, y) where {T} = Point{2, T}(Vec{2, T}(x, y))
Point{3, T}(x, y, z) where {T} = Point{3, T}(Vec{3, T}(x, y, z))
Point{1, T}(c::NTuple{1, T}) where {T} = Point{1, T}(Vec{1, T}(c[1]))
Point{2, T}(c::NTuple{2, T}) where {T} = Point{2, T}(Vec{2, T}(c[1], c[2]))
Point{3, T}(c::NTuple{3, T}) where {T} = Point{3, T}(Vec{3, T}(c[1], c[2], c[3]))
Point(c::Vec{1, T}) where {T} = Point{1, T}(c)
Point(c::Vec{2, T}) where {T} = Point{2, T}(c)
Point(c::Vec{3, T}) where {T} = Point{3, T}(c)
Point(c::NTuple{1, T}) where {T} = Point{1, T}(Vec{1, T}(c[1]))
Point(c::NTuple{2, T}) where {T} = Point{2, T}(Vec{2, T}(c[1], c[2]))
Point(c::NTuple{3, T}) where {T} = Point{3, T}(Vec{3, T}(c[1], c[2], c[3]))

# conversions
Base.convert(::Type{Point{2, T}}, P::Point{3}) where {T} = Point{2, T}(P[1], P[2])
Vec(P::Point{D, T}) where {D, T} = P.coords

# type aliases
const Point1  = Point{1, Float64}
const Point2  = Point{2, Float64}
const Point3  = Point{3, Float64}
const Point1f = Point{1, Float32}
const Point2f = Point{2, Float32}
const Point3f = Point{3, Float32}
const Point1B = Point{1, BigFloat}
const Point2B = Point{2, BigFloat}
const Point3B = Point{3, BigFloat}

# abstract array interface
Base.size(P::Point) = Base.size(P.coords)
Base.getindex(P::Point, i::Int) = Base.getindex(P.coords, i)
Base.IndexStyle(P::Point) = Base.IndexStyle(typeof(P.coords))

Base.zero(::Type{Point{D, T}}) where {D, T} = Point{D, T}(@SVector zeros(T, D))
nan(::Type{Point{D, T}}) where {D, T} = Point{D, T}(@SVector fill(T(NaN), D))

# Points separated by 1e-5 cm = 0.1 micron are treated the same.
const EPS_POINT = 1e-5
# Default coordinate for a point that is essentially infinitely far away.
# Used for when IEEE 754 may not be enforced, such as with fast math. 
const INF_POINT = 1e6

"""
    coordinates(p::Point)

Return the coordinates of the point with respect to the
canonical Euclidean basis.
"""
coordinates(p::Point) = p.coords

# Disallow point addition, multiplication, and division. 
# Leads to unwanted allocations, and is geometrically invalid.
Base.:+(A::Point, B::Point) = error("Point addition is not defined, nor should it be.")
Base.:*(A::Point, B) = error("Point multiplication is not defined, nor should it be.")
Base.:*(A, B::Point) = error("Point multiplication is not defined, nor should it be.")
Base.:/(A::Point, B) = error("Point division is not defined, nor should it be.")
Base.:/(A, B::Point) = error("Point division is not defined, nor should it be.")
"""
    -(A::Point, B::Point)

Return the [`Vec`](@ref) displacement from point `B` to point `A`.
"""
Base.:-(A::Point, B::Point) = coordinates(A) - coordinates(B)

"""
    +(A::Point, v::Vec)

Return the point at the end of the vector `v` placed
at a reference (or start) point `A`.
"""
Base.:+(A::Point, v::Vec) = Point(coordinates(A) + v)

"""
    -(A::Point, v::Vec)

Return the point at the end of the vector `-v` placed
at a reference (or start) point `A`.
"""
Base.:-(A::Point, v::Vec) = Point(coordinates(A) - v)

"""
    isapprox(A::Point, B::Point)

Return if two points are approximately equal (less than EPS_POINT apart). 
"""
function Base.isapprox(A::Point{D, T}, B::Point{D, T}) where {D, T}
    return norm²(A - B) < T(EPS_POINT^2)
end

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
midpoint(A::Point, B::Point) = Point((coordinates(A) + coordinates(B)) / 2)

"""
    isCCW(A::Point{2}, B::Point{2}, C::Point{2})

If the triplet of 2-dimensional points is counter-clockwise oriented from A to B to C.
"""
isCCW(A::Point{2}, B::Point{2}, C::Point{2}) = 0 ≤ (B - A) × (C - A)

function Base.show(io::IO, point::Point)
    return print(io, coordinates(point).data)
end
