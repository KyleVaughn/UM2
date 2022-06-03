function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Point{D, T}}) where {D, T}
    return Point{D, T}(rand(rng, Vec{D, T}))
end

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Edge{P, N, T}}) where {P, N, T}
    return Edge{P, N, T}(rand(rng, Vec{N, T}))
end

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{AABox{D, T}}) where {D, T}
    p1 = rand(rng, Vec{D, T})
    p2 = rand(rng, Vec{D, T})
    return AABox{D, T}(Point{D, T}(min.(p1, p2)), Point{D, T}(max.(p1, p2)))
end
