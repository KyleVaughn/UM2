export ⊙, ⊘, norm² 

"""
    ⊙(A::AbstractArray, B::AbstractArray)

Hadamard product (element-wise multiplication) of two equal size arrays.
"""
⊙(A::AbstractArray, B::AbstractArray) = map(*, A, B)

"""
    ⊘(a::AbstractArray, b::AbstractArray)

Hadamard division (element-wise division) of two equal size arrays.
"""
⊘(A::AbstractArray, B::AbstractArray) = map(/, A, B)

"""
    inv(v::AbstractVector) 

Return the Samelson inverse of a. 

The returned vector, a⁻¹, satisfies a⁻¹ ⋅ a = 1.
"""
inv(a::AbstractVector) = inv(a ⋅ a) * a'

"""
    norm²(v::AbstractVector)

Return the squared 2-norm of the vector.
"""
norm²(v::AbstractVector) = v ⋅ v
