# Methods to extend SVector functionality
@inline *(a::SVector, b::SVector) = map(*, a, b)
@inline /(a::SVector, b::SVector) = map(/, a, b)
@inline norm²(a::SVector) = a ⋅ a
@inline distance(a::SVector, b::SVector) = norm(a - b) 
@inline inv(a::SVector) = inv(a ⋅ a) * a' # Samelson inverse
