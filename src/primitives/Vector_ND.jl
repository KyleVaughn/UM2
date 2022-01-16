# Aliases for N-dimensional static vectors
const Vector_2D = SVector{2}
const Vector_3D = SVector{3}

# Methods
# -------------------------------------------------------------------------------------------------
@inline norm²(v::SVector) = v ⋅ v 
