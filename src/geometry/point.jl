export Point, Point1, Point2, Point3, Point1f, Point2f, Point3f
export distance, distance², coordinates, isCCW, midpoint, nan

"""
    Point{Dim, T}

A point in `Dim`-dimensional space with coordinates of type `T`.

The coordinates of the point are given with respect to the canonical
Euclidean basis, and `Integer` coordinates are converted to `Float64`.

## Examples

```julia
# 2D points
A = Point(0.0, 1.0) # double precision as expected
B = Point(0f0, 1f0) # single precision as expected
C = Point(0, 0) # Integer is converted to Float64 by design
D = Point2(0, 1) # explicitly ask for double precision
E = Point2f(0, 1) # explicitly ask for single precision

# 3D points
F = Point(1.0, 2.0, 3.0) # double precision as expected
G = Point(1f0, 2f0, 3f0) # single precision as expected
H = Point(1, 2, 3) # Integer is converted to Float64 by design
I = Point3(1, 2, 3) # explicitly ask for double precision
J = Point3f(1, 2, 3) # explicitly ask for single precision
```

### Notes

- Type aliases are `Point1`, `Point2`, `Point3`, `Point1f`, `Point2f`, `Point3f`
- `Integer` coordinates are not supported because most geometric processing
  algorithms assume a continuous space. The conversion to `Float64` avoids
  `InexactError` and other unexpected results.
"""
struct Point{Dim,T}
    coords::SVector{Dim,T}
    Point{Dim,T}(coords::SVector{Dim,T}) where {Dim,T} = new{Dim,T}(coords)
    Point{Dim,T}(coords::SVector{Dim,T}) where {Dim,T<:Integer} = new{Dim,Float64}(coords)
end

# constructors
Point{Dim,T}(coords...) where {Dim,T} = Point{Dim,T}(SVector{Dim,T}(coords...))
Point(coords::SVector{Dim,T}) where {Dim,T} = Point{Dim,T}(coords)
Point(coords::AbstractVector{T}) where {T} = Point{length(coords),T}(coords)
Point(coords...) = Point(SVector(coords...))

# conversions
convert(::Type{Point{Dim,T}}, coords) where {Dim,T} = Point{Dim,T}(coords)
convert(::Type{Point{Dim,T}}, p::Point) where {Dim,T} = Point{Dim,T}(p.coords)

# type aliases
const Point1  = Point{1,Float64}
const Point2  = Point{2,Float64}
const Point3  = Point{3,Float64}
const Point1f = Point{1,Float32}
const Point2f = Point{2,Float32}
const Point3f = Point{3,Float32}

# broadcast behavior
Base.broadcastable(p::Point) = Ref(p)

Base.@propagate_inbounds function Base.getindex(p::Point, i::Integer)
    getfield(p, :coords)[i]
end

function Base.getproperty(p::Point, sym::Symbol)
    if sym === :x
        return p.coords[1]
    elseif sym === :y
        return p.coords[2]
    elseif sym === :z
        return p.coords[3]
    else # fallback to getfield
        return getfield(p, sym)
    end
end

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
    +(A::Point, B::Point)

A convenience function that returns the [`Vec`](@ref) which satisfies (`B` - O) + (`A` - O),
where O is the origin.
"""
+(A::Point, B::Point) = A.coords + B.coords


"""
    +(A::Point, v::Vec)
    +(v::Vec, A::Point)

Return the point at the end of the vector `v` placed
at a reference (or start) point `A`.
"""
+(A::Point, v::Vec) = Point(A.coords + v)
+(v::Vec, A::Point) = A + v

"""
    -(A::Point, v::Vec)
    -(v::Vec, A::Point)

Return the point at the end of the vector `-v` placed
at a reference (or start) point `A`.
"""
-(A::Point, v::Vec) = Point(A.coords - v)
-(v::Vec, A::Point) = A - v

Base.isapprox(A::Point, B::Point; kwargs...) = isapprox(A.coords, B.coords; kwargs...)

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
  print(io, "Point$(Tuple(point.coords))")
end
