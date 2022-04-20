export edges

function edges(p::Polygon{N}) where {N}
    return map(i->begin
                    id₁ = (i - 1) % N + 1
                    id₂ = i % N + 1
                    return LineSegment(p[id₁], p[id₂])
                  end,
                  StaticArrays.sacollect(SVector{N,Int}, i for i = 1:N)
               )
end

function edges(p::QuadraticPolygon{N}) where {N}
    M = N ÷ 2
    return map(i->begin
                    id₁ = (i - 1) % M + 1
                    id₂ = i % M + 1
                    id₃ = i + M
                    return QuadraticSegment(p[id₁], p[id₂], p[id₃])
                  end,
                  StaticArrays.sacollect(SVector{M,Int}, i for i = 1:M)
               )
end
