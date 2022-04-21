export edges

@generated function edges(p::Polygon{N,T}) where {N,T}
    exprs = [begin
                id₁ = (i - 1) % N + 1
                id₂ = i % N + 1
                :(LineSegment(p[$id₁], p[$id₂]))
             end
             for i in 1:N
            ]
    return quote
        Base.@_inline_meta
        @inbounds return Vec{$N, LineSegment{$T}}(tuple($(exprs...)))
    end
end


@generated function edges(p::QuadraticPolygon{N,T}) where {N,T}
    M = N ÷ 2
    exprs = [begin
                id₁ = (i - 1) % M + 1
                id₂ = i % M + 1
                id₃ = i + M
                :(QuadraticSegment(p[$id₁], p[$id₂], p[$id₃]))
             end
             for i in 1:M
            ]
    return quote
        Base.@_inline_meta
        @inbounds return Vec{$M, QuadraticSegment{$T}}(tuple($(exprs...)))
    end
end
