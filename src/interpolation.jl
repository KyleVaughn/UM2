@inline (l::LineSegment)(r) = Point(l.𝘅₁.coord + r*l.𝘂)

# Note: 𝗾(0) = 𝘅₁, 𝗾(1) = 𝘅₂, 𝗾(1/2) = 𝘅₃
(q::QuadraticSegment)(r) = Point(((2r-1)*(r-1))q.𝘅₁ + (r*(2r-1))q.𝘅₂ + (4r*(1-r))q.𝘅₃)

