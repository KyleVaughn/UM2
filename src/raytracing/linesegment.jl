export LineSegment
export vertices 

"""
    LineSegment(P₁::Point{Dim,T}, 𝘂::Vec{Dim,T})
    LineSegment(P₁::Point{Dim,T}, P₂::Point{Dim,T})

A parametric `LineSegment` in `Dim`-dimensional space with elements of type `T`.

The segment starts at point `P₁`, ends at point `P₂`, and satisfies
the equation L(r) = P₁ + r𝘂, where 𝘂 = P₂ - P₁ and r ∈ [0, 1].

### Notes

- L(0) = P₁, L(1) = P₂
"""
struct LineSegment{Dim,T}
    P₁::Point{Dim,T}  
    𝘂::Vec{Dim,T}
end

# constructors
LineSegment(P₁::Point{Dim,T}, P₂::Point{Dim,T}) where {Dim,T} = LineSegment{Dim,T}(P₁, P₂ - P₁) 
LineSegment(v::Vec{2, Point{Dim,T}}) where {Dim,T} = LineSegment{Dim,T}(v[1], v[2] - v[1])

function Base.getproperty(l::LineSegment, sym::Symbol)
    if sym === :P₂
        return l.P₁ + l.𝘂
    else # fallback to getfield
        return getfield(l, sym)
    end
end

vertices(l::LineSegment) = (l.P₁, l.P₂)

function Base.show(io::IO, l::LineSegment)
    print(io, "LineSegment$(vertices(l))")
end
