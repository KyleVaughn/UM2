"""
    QuadraticSegment(SVector{3, Point{Dim, T}})
    QuadraticSegment(p₁::Point{Dim, T}, p₂::Point{Dim, T}, p₃::Point{Dim, T})

Construct a parametric quadratic segment in `Dim`-dimensional space that starts at 
point 𝘅₁, passes through 𝘅₃ at r=1//2, and ends at point 𝘅₂. The segment satisfies 
𝗾(r) = (2r-1)(r-1)𝘅₁ + r(2r-1)𝘅₂ + 4r(1-r)𝘅₃ where r ∈ [0, 1]. Equivalently, 
𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁, where 𝘂 = 2(𝘅₁ + 𝘅₂ - 2𝘅₃) and 𝘃 = -(3𝘅₁ + 𝘅₂ - 4𝘅₃).
                ___𝘅₃___
           ____/         ___
       ___/                  \
    __/                       𝘅₂
  _/
 /
𝘅₁
"""
struct QuadraticSegment{Dim, T} <:Edge{Dim, 2, T}
    points::SVector{3, Point{Dim, T}}
end

const QuadraticSegment2D = QuadraticSegment{2}
const QuadraticSegment3D = QuadraticSegment{3}

Base.@propagate_inbounds function Base.getindex(q::QuadraticSegment, i::Integer)
    getfield(q, :points)[i]
end

# Easily fetch 𝘂, 𝘃, in 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
function Base.getproperty(q::QuadraticSegment, sym::Symbol)
    if sym === :𝘂
        return 2(q[1] + q[2] - 2q[3])
    elseif sym === :𝘃
        return 4q[3] - 3q[1] - q[2]
    elseif sym === :𝘅₁
        return q[1] 
    elseif sym === :𝘅₂
        return q[2] 
    elseif sym === :𝘅₃
        return q[3] 
    else # fallback to getfield
        return getfield(q, sym)
    end
end

function QuadraticSegment(p₁::Point{Dim, T}, 
                          p₂::Point{Dim, T}, 
                          p₃::Point{Dim, T}) where {Dim, T}
    return QuadraticSegment{Dim, T}(SVector{3, Point{Dim, T}}(p₁, p₂, p₃))
end
function QuadraticSegment{Dim}(p₁::Point{Dim, T}, 
                               p₂::Point{Dim, T}, 
                               p₃::Point{Dim, T}) where {Dim, T}
    return QuadraticSegment{Dim, T}(SVector{3, Point{Dim, T}}(p₁, p₂, p₃))
end

# Note: 𝗾(0) = 𝘅₁, 𝗾(1) = 𝘅₂, 𝗾(1/2) = 𝘅₃
(q::QuadraticSegment)(r) = Point(((2r-1)*(r-1))q.𝘅₁ + (r*(2r-1))q.𝘅₂ + (4r*(1-r))q.𝘅₃)

# Return the Jacobian of q, evalutated at r
# 𝗾′(r) = 2r𝘂 + 𝘃, which is simplified to below.
jacobian(q::QuadraticSegment, r) = (4r - 3)*(q.𝘅₁ - q.𝘅₃) + (4r - 1)*(q.𝘅₂ - q.𝘅₃) 

# If the line is straight, 𝘅₃ - 𝘅₁ = c(𝘅₂ - 𝘅₁) where c ∈ (0, 1), hence
# (𝘅₃ - 𝘅₁) × (𝘅₂ - 𝘅₁) = 𝟬
function isstraight(q::QuadraticSegment2D)
    return abs((q.𝘅₃ - q.𝘅₁) × (q.𝘅₂ - q.𝘅₁)) < 1e-5
end
function isstraight(q::QuadraticSegment3D)
    return norm²((q.𝘅₃ - q.𝘅₁) × (q.𝘅₂ - q.𝘅₁)) < 1e-10
end
