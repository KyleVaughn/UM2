function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Point{Dim,T}}) where {Dim,T}
    return Point{Dim,T}(rand(rng, Vec{Dim,T}))
end

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{Edge{P,N,T}}) where {P,N,T}
    return Edge{P,N,T}(rand(rng, Vec{N,T}))
end

function Random.rand(rng::AbstractRNG, ::Random.SamplerType{AABox{Dim,T}}) where {Dim,T}
    p1 = rand(rng, Vec{Dim,T}) 
    p2 = rand(rng, Vec{Dim,T})
    return AABox{Dim,T}(Point{Dim,T}(min.(p1, p2)), Point{Dim,T}(max.(p1, p2)))  
end
