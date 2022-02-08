# A Dim-dimensional point
struct Point{Dim, T}
    coord::SVector{Dim, T}
end

const Point2D = Point{2}

Base.broadcastable(p::Point) = Ref(p)
Base.@propagate_inbounds function Base.getindex(p::Point, i::Integer)
    getfield(p, :coord)[i]
end
# All branches but the correct one are pruned by the compiler, so this is fast
# when called inside a function.
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
Point{Dim}(v::SVector{Dim, T}) where {Dim, T}= Point{Dim,T}(v)
Point{Dim, T}(x...) where {Dim, T}= Point{Dim, T}(SVector{Dim, T}(x))
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
Base.zero(::Type{Point{Dim, T}}) where {Dim, T} = Point{Dim, T}(@SVector zeros(T, Dim))
# Random point in the Dim-dimensional unit hypercube
Base.rand(::Type{Point{Dim, T}}) where {Dim, T} = Point{Dim, T}(rand(SVector{Dim, T}))
function Base.rand(::Type{Point{Dim, T}}, num_points::Int64) where {Dim, T}
    return [ Point{Dim, T}(rand(SVector{Dim, T})) for i = 1:num_points ]
end

# Sort points based on their distance prom another point.
# Default algorithm is quicksort. If the vector is less than 20 elements, insertion sort
# is used instead.
defalg(v::Vector{<:Point}) = Base.Sort.QuickSort
function sort!(p::Point, v::Vector{<:Point}; 
               alg::Base.Sort.Algorithm=defalg(v), order::Base.Ordering=Base.Forward)
    sort!(p, v, firstindex(v), lastindex(v), alg, order)
end

function sort(p::Point, v::Vector{<:Point};
               alg::Base.Sort.Algorithm=defalg(v), order::Base.Ordering=Base.Forward)
    v2 = similar(v)
    @. v2 = v
    sort!(p, v2, firstindex(v2), lastindex(v2), alg, order)
    return v2
end

function sort!(p::Point, v::Vector{<:Point}, lo::Integer, hi::Integer,
               ::Base.Sort.InsertionSortAlg, o::Base.Ordering)
    @inbounds for i ∈ lo+1:hi
        j = i
        d = distance²(p, v[i])
        pt = v[i]
        while j > lo
            if Base.lt(o, d, distance²(p, v[j-1]))
                v[j] = v[j-1]
                j -= 1
                continue
            end
            break
        end
        v[j] = pt
    end
    return v
end

@inline function selectpivot!(p::Point, v::Vector{<:Point}, lo::Integer, hi::Integer,
    o::Base.Ordering)
    @inbounds begin
        mi = Base.Sort.midpoint(lo, hi)

        # sort v[mi] <= v[lo] <= v[hi] such that the pivot is immediately in place
        if Base.lt(o, distance²(p, v[lo]), distance²(p, v[mi]))
            v[mi], v[lo] = v[lo], v[mi]
        end

        if Base.lt(o, distance²(p, v[hi]), distance²(p, v[lo]))
            if Base.lt(o, distance²(p, v[hi]), distance²(p, v[mi]))
                v[hi], v[lo], v[mi] = v[lo], v[mi], v[hi]
            else
                v[hi], v[lo] = v[lo], v[hi]
            end
        end

        # return the pivot
        return v[lo]
    end
end

function partition!(p::Point, v::Vector{<:Point}, lo::Integer, hi::Integer, o::Base.Ordering)
    pivot = selectpivot!(p, v, lo, hi, o)
    d = distance²(p, pivot)
    # pivot == v[lo], v[hi] > pivot
    i, j = lo, hi
    @inbounds while true
        i += 1; j -= 1
        while Base.lt(o, distance²(p, v[i]), d); i += 1; end;
        while Base.lt(o, d, distance²(p, v[j])); j -= 1; end;
        i >= j && break
        v[i], v[j] = v[j], v[i]
    end
    v[j], v[lo] = pivot, v[j]

    # v[j] == pivot
    # v[k] >= pivot for k > j
    # v[i] <= pivot for i < j
    return j
end

function sort!(p::Point, v::Vector{<:Point}, lo::Integer, hi::Integer,
               a::Base.Sort.QuickSortAlg, o::Base.Ordering)
    @inbounds while lo < hi
        if hi-lo ≤ Base.Sort.SMALL_THRESHOLD
            return sort!(p, v, lo, hi, Base.Sort.InsertionSort, o)
        end
        j = partition!(p, v, lo, hi, o)
        if j-lo < hi-j
            # recurse on the smaller chunk
            # this is necessary to preserve O(log(n))
            # stack space in the worst case (rather than O(n))
            lo < (j-1) && sort!(p, v, lo, j-1, a, o)
            lo = j+1
        else
            j+1 < hi && sort!(p, v, j+1, hi, a, o)
            hi = j-1
        end
    end
    return v
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
