export LineSegment

"""
    LineSegment(A::Point{Dim,T}, 𝘂::Vec{Dim,T})
    LineSegment(A::Point{Dim,T}, B::Point{Dim,T})

A parametric `LineSegment` in `Dim`-dimensional space with elements of type `T`.

The segment starts at point `A`, ends at point `B`, and satisfies
the equation L(r) = A + r𝘂, where 𝘂 = B - A and r ∈ [0, 1].

### Notes

- Type aliases are `LineSegment2` and `LineSegment3`.
"""
struct LineSegment{Dim,T}
    A::Point{Dim,T}  
    𝘂::Vec{Dim,T}
end

# constructors
LineSegment(A::Point{Dim,T}, B::Point{Dim,T}) where {Dim,T} = LineSegment{Dim,T}(A, B - A) 
LineSegment(v::Vec{2, Point{Dim,T}}) where {Dim,T} = LineSegment{Dim,T}(v[1], v[2] - v[1])

function Base.getproperty(l::LineSegment, sym::Symbol)
    if sym === :B
        return l.A + l.𝘂
    else # fallback to getfield
        return getfield(l, sym)
    end
end

function Base.show(io::IO, l::LineSegment)
    print(io, "LineSegment($(l.A), $(l.B))")
end
