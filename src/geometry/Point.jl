"""
    Point{Dim, T}(x...)

Construct a `Dim`-dimensional `Point` in Euclidian space, with data of type `T`. 
Constructors may drop the `Dim` and `T` parameters if they are inferrable from the
input (e.g. `Point(1,2,3)` constructs an `Point{3, Int64}`).
"""
struct Point{Dim, T}
    coord::SVector{Dim, T}
end

const Point1D = Point{1}
const Point2D = Point{2}
const Point3D = Point{3}

Base.broadcastable(p::Point) = Ref(p)

Base.@propagate_inbounds function Base.getindex(p::Point, i::Integer)
    getfield(p, :coord)[i]
end

function Base.getproperty(p::Point, sym::Symbol)
    if sym === :x
        return p.coord[1]
    elseif sym === :y
        return p.coord[2]
    elseif sym === :z
        return p.coord[3]
    else # fallback to getfield
        return getfield(p, sym)
    end
end

Point{Dim}(v::SVector{Dim, T}) where {Dim, T}= Point{Dim, T}(v)
Point{Dim, T}(x...) where {Dim, T}= Point{Dim, T}(SVector{Dim, T}(x))
Point{Dim}(x...) where {Dim}= Point(SVector(x))
Point(x...) = Point(SVector(x))

convert(::Type{P}, p::Point) where {P <: Point} = P(p.coord)
convert(::Type{P}, p::Point) where {P <: Point2D} = P(p[1], p[2])

zero(::Type{Point{Dim, T}}) where {Dim, T} = Point{Dim, T}(@SVector zeros(T, Dim))
nan(::Type{Point{Dim, T}}) where {Dim, T} = Point{Dim, T}(@SVector fill(T(NaN), Dim))

# Ensure that resulting types make sense if more functions are added.
# Points are not vectors! A negative point doesn't make sense!
# A - B ≠ A + (-B)
# See Affine Space or Euclidian Space wikipedia entries.
@inline +(p::Point, v::SVector) = Point(p.coord + v)
@inline +(v::SVector, p::Point) = Point(v + p.coord)

@inline -(p₁::Point, p₂::Point) = p₁.coord - p₂.coord
@inline -(p::Point, v::SVector) = Point(p.coord - v)

@inline ==(p::Point, v::Vector) = p.coord == v
@inline ≈(p₁::Point, p₂::Point) = distance²(p₁, p₂) < (1e-5)^2 # 100 nm

@inline distance(p₁::Point, p₂::Point) = norm(p₁ - p₂)
@inline distance²(p₁::Point, p₂::Point) = norm²(p₁ - p₂)
@inline midpoint(p₁::Point, p₂::Point) = Point((p₁.coord + p₂.coord)/2)

"""
    isCCW(p₁::Point2D, p₂::Point2D, p₃::Point2D)

If the triplet of points is counter-clockwise oriented from p₁ to p₂ to p₃.
"""
@inline isCCW(p₁::Point2D, p₂::Point2D, p₃::Point2D) = 0 ≤ (p₂ - p₁) × (p₃ - p₁)
