# Methods to extend SVector functionality
@inline norm²(v::SVector) = v ⋅ v
@inline normalize(v::SVector) = v/norm(v) 
@inline distance(v₁::SVector, v₂::SVector) = norm(v₁ - v₂) 
@inline inv(v::SVector) = v'/(v ⋅ v) # Samelson inverse
@inline *(v₁::SVector, v₂::SVector) = v₁ .* v₂
@inline /(v₁::SVector, v₂::SVector) = v₁ ./ v₂
