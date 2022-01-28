# An Dim-dimensional point
struct Point{Dim, T}
    coord::SVector{Dim, T}
end

const Point2D = Point{2}

Base.broadcastable(p::Point) = Ref(p)
Base.@propagate_inbounds function Base.getindex(p::Point, i::Integer)
    getfield(p, :coord)[i]
end
# all branches but the correct one are pruned by the compiler, so this is fast.
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

# Constructors
# ---------------------------------------------------------------------------------------------
Point{Dim,T}(x...) where {Dim,T}= Point{Dim,T}(SVector{Dim,T}(x))
Point{Dim}(x...) where {Dim}= Point(SVector(x))
Point(x...) = Point(SVector(x))

# Operators
# ---------------------------------------------------------------------------------------------
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

@inline /(n::Number, p::Point) = Point(n / p.coord) 
@inline /(p::Point, n::Number) = Point(p.coord / n)

@inline ⋅(p₁::Point, p₂::Point) = dot(p₁.coord, p₂.coord)
@inline ×(p₁::Point, p₂::Point) = cross(p₁.coord, p₂.coord)
@inline ==(p::Point, v::Vector) = p.coord == v
@inline ≈(p₁::Point, p₂::Point) = distance²(p₁, p₂) < (1e-5)^2 # 100 nm

# Methods
# ---------------------------------------------------------------------------------------------
@inline distance(p₁::Point, p₂::Point) = norm(p₁ - p₂)
@inline distance(p₁::Point, v::Vector) = norm(p - v)
@inline distance(v::Vector, p::Point) = norm(v - p)
@inline distance²(p₁::Point, p₂::Point) = norm²(p₁ - p₂)
@inline midpoint(p₁::Point, p₂::Point) = (p₁ + p₂)/2
@inline norm(p::Point) = √(p.coord ⋅ p.coord)
@inline norm²(p::Point) = p.coord ⋅ p.coord
Base.zero(::Type{Point{Dim,T}}) where {Dim,T} = Point{Dim,T}(@SVector zeros(T, Dim))
# Random point in the Dim-dimensional unit hypercube
Base.rand(::Type{Point{Dim,T}}) where {Dim,T} = Point{Dim,T}(rand(SVector{Dim,T}))
function Base.rand(::Type{Point{Dim,T}}, num_points::Int64) where {Dim,T}
    return [ Point{Dim,T}(rand(SVector{Dim,T})) for i = 1:num_points ]
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(T::Type{<:Scatter}, p::Point)
        return convert_arguments(T, p.coord)
    end

    function convert_arguments(T::Type{<:Scatter}, P::Vector{<:Point})
        return convert_arguments(T, [p.coord for p in P])
    end

    function convert_arguments(T::Type{<:LineSegments}, p::Point)
        return convert_arguments(T, p.coord)
    end

    function convert_arguments(T::Type{<:LineSegments}, P::Vector{<:Point})
        return convert_arguments(T, [p.coord for p in P])
    end
end
