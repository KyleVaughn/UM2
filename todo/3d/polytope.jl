# -- Accessors --

vertices(p::Polytope) = p.vertices

peaks(p::Polytope{3}) = vertices(p)

ridges(p::Polytope{2}) = vertices(p)
ridges(p::Polytope{3}) = edges(p)

facets(p::Polytope{1}) = vertices(p)
facets(p::Polytope{2}) = edges(p)
facets(p::Polytope{3}) = faces(p)


## If we think of the polytopes as sets, p₁ ∩ p₂ = p₁ and p₁ ∩ p₂ = p₂ implies p₁ = p₂
#function Base.:(==)(l₁::LineSegment{T}, l₂::LineSegment{T}) where {T}
#    return (l₁[1] === l₂[1] && l₁[2] === l₂[2]) ||
#           (l₁[1] === l₂[2] && l₁[2] === l₂[1])
#end
#Base.:(==)(t₁::Triangle, t₂::Triangle) = return all(v -> v ∈ t₂.vertices, t₁.vertices)
#Base.:(==)(t₁::Tetrahedron, t₂::Tetrahedron) = return all(v -> v ∈ t₂.vertices, t₁.vertices)
#function Base.:(==)(q₁::QuadraticSegment{T}, q₂::QuadraticSegment{T}) where {T}
#    return q₁[3] === q₂[3] &&
#           (q₁[1] === q₂[1] && q₁[2] === q₂[2]) ||
#           (q₁[1] === q₂[2] && q₁[2] === q₂[1])
#end

#isstraight(::LineSegment) = true
#
#"""
#    isstraight(q::QuadraticSegment)
#
#Return if the quadratic segment is effectively straight.
#(If P₃ is at most EPS_POINT distance from LineSegment(P₁,P₂))
#"""
#function isstraight(q::QuadraticSegment{T}) where {T <: Point}
#    # Project P₃ onto the line from P₁ to P₂, call it P₄
#    𝘃₁₃ = q[3] - q[1]
#    𝘃₁₂ = q[2] - q[1]
#    v₁₂ = norm²(𝘃₁₂)
#    𝘃₁₄ = (𝘃₁₃ ⋅ 𝘃₁₂) * inv(v₁₂) * 𝘃₁₂
#    # Determine the distance from P₃ to P₄ (P₄ - P₃ = P₁ + 𝘃₁₄ - P₃ = 𝘃₁₄ - 𝘃₁₃)
#    d² = norm²(𝘃₁₄ - 𝘃₁₃)
#    return d² < T(EPS_POINT^2)
#end
