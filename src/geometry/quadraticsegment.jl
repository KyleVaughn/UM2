export QuadraticSegment
export isstraight, points

"""
    QuadraticSegment(Vec{3, Point{Dim,T}})
    QuadraticSegment(P₁::Point{Dim,T}, P₂::Point{Dim,T}, P₃::Point{Dim,T})

A parametric `QuadraticSegment` in `Dim`-dimensional space with elements of type `T`.

The segment starts at point `P₁`, ends at `P₂`, and passes through `P₃`.
The segment satisfies the equation Q(r) = P₁ + r𝘂 + r²𝘃, where 
- 𝘂 = -(3𝗽₁ + 𝗽₂ - 4𝗽₃) and 𝘃 = 2(𝗽₁ + 𝗽₂ - 2𝗽₃),
- 𝗽ᵢ = Pᵢ - O, for i = 1:3, where O is the origin, 
- r ∈ [0, 1]

### Notes

- Equivalently, Q(r) = (2r-1)(r-1)𝗽₁ + r(2r-1)𝗽₂ + 4r(1-r)𝗽₃. 
- Q(0) = P₁, Q(1) = P₂, Q(1/2) = P₃
"""
struct QuadraticSegment{Dim,T}
    P₁::Point{Dim,T}
    𝘂::Vec{Dim,T}
    𝘃::Vec{Dim,T}
end

# constructors
function QuadraticSegment(P₁::Point{Dim,T}, 
                          P₂::Point{Dim,T}, 
                          P₃::Point{Dim,T}) where {Dim, T}
    𝗮 = P₁ - P₃
    𝗯 = P₂ - P₃
    𝘂 = -3𝗮 - 𝗯
    𝘃 = 2(𝗮 + 𝗯)
    return QuadraticSegment{Dim, T}(P₁, 𝘂, 𝘃)
end

Base.@propagate_inbounds function Base.getindex(q::QuadraticSegment, i::Integer)
    getfield(q, :points)[i]
end

function Base.getproperty(q::QuadraticSegment, sym::Symbol)
    if sym === :P₂
        return q.P₁ + q.𝘂 + q.𝘃
    elseif sym === :P₃
        return q.P₁ + q.𝘂/2 + q.𝘃/4
    else # fallback to getfield
        return getfield(q, sym)
    end
end

points(q::QuadraticSegment) = (q.P₁, q.P₂, q.P₃)

function isstraight(q::QuadraticSegment)
    return norm²(q.𝘃) < 1e-6
end

function Base.show(io::IO, q::QuadraticSegment)
    print(io, "QuadraticSegment$(points(q))")
end
