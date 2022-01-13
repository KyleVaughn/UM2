# An N-dimensional point
struct Point{N,T} <: StaticVector{N,T}
    data::NTuple{N,T}

    function Point{N,T}(x::NTuple{N,T}) where {N,T}
        return new{N,T}(x)
    end

    function Point{N,T}(x::NTuple{N,Any}) where {N,T}
        return new{N,T}(StaticArrays.convert_ntuple(T, x))
    end
end

const Point_2D = Point{2}
const Point_3D = Point{3}

# NOTE: This is pretty much an exact copy of the Point in GeometryBasics. 
# (https://github.com/JuliaGeometry/GeometryBasics.jl/blob/master/src/fixed_arrays.jl)

# Constructors & Conversions
# -------------------------------------------------------------------------------------------------
# Construct from vector with known size 
function Point{N}(x::AbstractVector{T}) where {N,T}
    @assert N <= length(x)
    return Point{N,T}(ntuple(i -> x[i], Val(N)))
end

# Construct from vector with known size, and convert to type T₁
function Point{N,T₁}(x::AbstractVector{T₂}) where {N,T₁,T₂}
    @assert N <= length(x)
    return Point{N,T₁}(ntuple(i -> convert(T₁, x[i]), Val(N)))
end

# Construct from a Tuple
function Point(x::T) where {N,T <: Tuple{Vararg{Any,N}}}
    return Point{N,StaticArrays.promote_tuple_eltype(T)}(x)
end
 
# Construct from a Tuple with known size 
function Point{N}(x::T) where {N,T <: Tuple}
    return Point{N,StaticArrays.promote_tuple_eltype(T)}(x)
end
 
# Construct from a Point
@generated function (::Type{Point{N,T}})(p::Point) where {N,T}
    idx = [:(p[$i]) for i in 1:N]
    return quote
        $(Point){N,T}($(idx...))
    end
end

# Convert from a Point
@generated function Base.convert(::Type{Point{N,T}}, p::Point) where {N,T}
    idx = [:(p[$i]) for i in 1:N]
    return quote
        $(Point){N,T}($(idx...))
    end
end

# Convert Tuple to Point
function Base.convert(::Type{Point{N,T}}, x::NTuple{N,T}) where {N,T}
    return Point{N,T}(x)
end
function Base.convert(::Type{Point{N,T}}, x::Tuple) where {N,T}
    return Point{N,T}(convert(NTuple{N,T}, x))
end

# Base (and similar_type)
# -------------------------------------------------------------------------------------------------
@generated function StaticArrays.similar_type(::Type{SV}, ::Type{T},
                                              s::Size{N}) where {SV <: Point,T,N}
    return if length(N) === 1
        Point{N[1],T}
    else
        StaticArrays.default_similar_type(T, s(), Val{length(N)})
    end
end
function Base.broadcasted(f, a::AbstractArray{T}, b::Point) where {T <: Point}
    return Base.broadcasted(f, a, (b,))
end
Base.@propagate_inbounds function Base.getindex(p::Point{N,T}, i::Int) where {N,T}
    return p.data[i]
end
Base.Tuple(p::Point) = p.data

# Methods
# -------------------------------------------------------------------------------------------------
@inline distance(𝐩₁::Point, 𝐩₂::Point) = norm(𝐩₁ - 𝐩₂)
@inline distance²(𝐩₁::Point, 𝐩₂::Point) = norm²(𝐩₁ - 𝐩₂)
@inline Base.isapprox(𝐩₁::Point, 𝐩₂::Point) = distance²(𝐩₁, 𝐩₂) < (1e-5)^2 # 100 nm
@inline midpoint(𝐩₁::Point, 𝐩₂::Point) = (𝐩₁ + 𝐩₂)/2
@inline norm²(𝐩::Point) = 𝐩 ⋅ 𝐩

# Sort points based on their distance from a given point
sortpoints(p::Point, points::Vector{<:Point}) = points[sortperm(distance².(Ref(p), points))]
function sortpoints!(p::Point_2D, points::Vector{<:Point_2D})
    permute!(points, sortperm(distance².(Ref(p), points)))
    return nothing
end
