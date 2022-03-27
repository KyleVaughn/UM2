# Methods to extend SVector functionality

"""
    *(a::SVector, b::SVector)

The element-wise multiplication of two equal length vectors.

# Examples
```jldoctest
julia> SVector(1, 2, 3) * SVector(2, 3, 4)
3-element SVector{3, Int64} with indices SOneTo(3):
  2
  6
 12
```
"""
@inline *(a::SVector, b::SVector) = map(*, a, b)

"""
    /(a::SVector, b::SVector)

The element-wise division of two equal length vectors.

# Examples
```jldoctest
julia> SVector(8, 4, 2) / SVector(4, 2, 1)
3-element SVector{3, Float64} with indices SOneTo(3):
 2.0
 2.0
 2.0
```
"""
@inline /(a::SVector, b::SVector) = map(/, a, b)

"""
    norm²(a::SVector{N, T}) where {N, T<:Real} 

The squared value of 2-norm of the real-valued vector `a`, ‖a‖².
"""
@inline norm²(a::SVector{N, T}) where {N, T<:Real} = a ⋅ a

"""
    distance(a::SVector, b::SVector)

The Euclidian distance from point `a` to point `b`.
"""
@inline distance(a::SVector, b::SVector) = norm(a - b) 

"""
    inv(a::SVector) 

Return the Samelson inverse of a. 

The returned vector, a⁻¹, satisfies a⁻¹ ⋅ a = 1.
"""
@inline inv(a::SVector) = inv(a ⋅ a) * a'
