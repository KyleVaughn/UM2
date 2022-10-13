function Base.intersect(R::Ray2{T}, aab::AABox2{T}) where {T <: AbstractFloat} 
    𝗱⁻¹= 1 / R.direction
    𝘁₁ = 𝗱⁻¹*(aab.minima - R.origin)
    𝘁₂ = 𝗱⁻¹*(aab.maxima - R.origin)
    tmin = maximum(min.(𝘁₁.coord, 𝘁₂.coord))
    tmax = minimum(max.(𝘁₁.coord, 𝘁₂.coord))
    if tmin ≤ tmax 
        return (tmin, tmax)
    else
        return (T(INF_POINT), T(INF_POINT))
    end
end
