"""
    Point{Dim, T}(x...)

Construct a `Dim`-dimensional point with data of type `T`. 
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

Base.zero(::Type{Point{Dim, T}}) where {Dim, T} = Point{Dim, T}(@SVector zeros(T, Dim))
nan(::Type{Point{Dim, T}}) where {Dim, T} = Point{Dim, T}(@SVector fill(T(NaN), Dim))

# Operators of the form X(Point, Number) perform X.(p.coord, n)
# Operators of the form X(Point, Point) or X(Point, SVector) perform element-wise 
# operations, except in the case of ⋅, ×, and ≈.
@inline +(p::Point, n::Number) = Point(p.coord .+ n)
@inline +(n::Number, p::Point) = Point(n .+ p.coord)
@inline +(p₁::Point, p₂::Point) = p₁.coord + p₂.coord
@inline +(p::Point, v::SVector) = p.coord + v
@inline +(v::SVector, p::Point) = v + p.coord

@inline -(p₁::Point, p₂::Point) = p₁.coord - p₂.coord
@inline -(p::Point, v::SVector) = p.coord - v
@inline -(v::SVector, p::Point) = v - p.coord
@inline -(p::Point) = Point(-p.coord)
@inline -(p::Point, n::Number) = Point(p.coord .- n)
@inline -(n::Number, p::Point) = Point(n .- p.coord)

@inline *(n::Number, p::Point) = Point(n * p.coord) 
@inline *(p::Point, n::Number) = Point(p.coord * n)
@inline *(p₁::Point, p₂::Point) = Point(p₁.coord .* p₂.coord)

@inline /(n::Number, p::Point) = Point(n / p.coord) 
@inline /(p::Point, n::Number) = Point(p.coord / n)
@inline /(p₁::Point, p₂::Point) = Point(p₁.coord ./ p₂.coord)

@inline ⋅(p₁::Point, p₂::Point) = dot(p₁.coord, p₂.coord)
@inline ×(p₁::Point, p₂::Point) = cross(p₁.coord, p₂.coord)
@inline ==(p::Point, v::Vector) = p.coord == v
@inline ≈(p₁::Point, p₂::Point) = distance²(p₁, p₂) < (1e-5)^2 # 100 nm

@inline distance(p₁::Point, p₂::Point) = norm(p₁ - p₂)
@inline distance(p₁::Point, v::Vector) = norm(p - v)
@inline distance(v::Vector, p::Point) = norm(v - p)
@inline distance²(p₁::Point, p₂::Point) = norm²(p₁ - p₂)
@inline midpoint(p₁::Point, p₂::Point) = (p₁ + p₂)/2
@inline norm(p::Point) = √(p.coord ⋅ p.coord)
@inline norm²(p::Point) = p.coord ⋅ p.coord
