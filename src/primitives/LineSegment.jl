"""
    LineSegment(𝘅₁::Point{Dim, T}, 𝘂::SVector{Dim, T})
    LineSegment(𝘅₁::Point{Dim, T}, 𝘅₂::Point{Dim, T})

Construct a parametric `LineSegment` in `Dim`-dimensional space that starts at 
point 𝘅₁ and ends at point 𝘅₂. The line satisfies the equation 𝗹(r) = 𝘅₁ + r𝘂,
where 𝘂 = 𝘅₂ - 𝘅₁ and r ∈ [0, 1].
"""
struct LineSegment{Dim, T} <:Edge{Dim, 1, T}
    𝘅₁::Point{Dim, T} 
    𝘂::SVector{Dim, T} # Store 𝘂 instead of 𝘅₂, since 𝘂 is needed much more often.
end

const LineSegment2D = LineSegment{2}
const LineSegment3D = LineSegment{3}

function Base.getproperty(l::LineSegment, sym::Symbol)
    if sym === :𝘅₂
        return Point(l.𝘅₁ + l.𝘂)
    else # fallback to getfield
        return getfield(l, sym)
    end
end

# Construct from Points
LineSegment{Dim, T}(𝘅₁::Point{Dim, T}, 𝘅₂::Point{Dim, T}) where {Dim, T} = 
    LineSegment{Dim, T}(𝘅₁, 𝘅₂ - 𝘅₁) 

LineSegment{Dim}(𝘅₁::Point{Dim, T}, 𝘅₂::Point{Dim, T}) where {Dim, T} = 
    LineSegment{Dim, T}(𝘅₁, 𝘅₂ - 𝘅₁) 

LineSegment(𝘅₁::Point{Dim, T}, 𝘅₂::Point{Dim, T}) where {Dim, T} = 
    LineSegment{Dim, T}(𝘅₁, 𝘅₂ - 𝘅₁) 

# Construct from SVector of points
LineSegment{Dim, T}(pts::SVector{2, Point{Dim, T}}) where {Dim, T} = 
    LineSegment{Dim, T}(pts[1], pts[2] - pts[1]) 
LineSegment{Dim}(pts::SVector{2, Point{Dim, T}}) where {Dim, T} = 
    LineSegment{Dim, T}(pts[1], pts[2] - pts[1]) 
LineSegment(pts::SVector{2, Point{Dim, T}}) where {Dim, T} = 
    LineSegment{Dim, T}(pts[1], pts[2] - pts[1]) 
