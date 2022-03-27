# Additional linear algebra methods 

"""
    ⊙(a::AbstractArray, b::AbstractArray)

Hadamard product (element-wise multiplication) of two equal size arrays.
"""
@inline ⊙(a::SVector, b::SVector) = map(*, a, b)

"""
    ⊘(a::AbstractArray, b::AbstractArray)

Hadamard division (element-wise division) of two equal size arrays.
"""
@inline ⊘(a::SVector, b::SVector) = map(/, a, b)

"""
    inv(a::SVector) 

Return the Samelson inverse of a. 

The returned vector, a⁻¹, satisfies a⁻¹ ⋅ a = 1.
"""
@inline inv(a::SVector) = inv(a ⋅ a) * a'

@inline norm²(a::SVector{N, T}) where {N, T<:Real} = a ⋅ a
