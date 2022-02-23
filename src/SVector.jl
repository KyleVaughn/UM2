# Methods to extend SVector functionality

# Aliases for N-dimensional static vectors
const Vector2D = SVector{2}
const Vector3D = SVector{3}

# Methods
# ---------------------------------------------------------------------------------------------
@inline norm²(v::SVector) = v ⋅ v
@inline normalize(v::SVector) = v/norm(v) 
@inline distance(v₁::SVector, v₂::SVector) = norm(v₁ - v₂) 
@inline inv(v::SVector) = v'/(v ⋅ v) # Samelson inverse
@inline *(v₁::SVector, v₂::SVector) = v₁ .* v₂
@inline /(v₁::SVector, v₂::SVector) = v₁ ./ v₂
