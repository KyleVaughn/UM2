export edges

# TODO: Figure out a more elegant/idiomatic way to do this
@generated function edges(p::Polygon{N}) where {N}
    edges_string = "Vec("
    for i ∈ 1:N
        id₁ = (i - 1) % N + 1
        id₂ = i % N + 1
        edges_string *= "LineSegment(p[$id₁], p[$id₂]), "
    end
    edges_string *= ")"
    return Meta.parse(edges_string)
end

@generated function edges(p::QuadraticPolygon{N}) where {N}
    M = N ÷ 2
    edges_string = "Vec("
    for i ∈ 1:M
        id₁ = (i - 1) % M + 1
        id₂ = i % M + 1
        id₃ = i + M
        edges_string *= "QuadraticSegment(p[$id₁], p[$id₂], p[$id₃]), "
    end
    edges_string *= ")"
    return Meta.parse(edges_string)
end
