@inline measure(aab::AABox2D) = Δx(aab) * Δy(aab)
@inline measure(aab::AABox3D) = Δx(aab) * Δy(aab) * Δz(aab)
@inline measure(l::LineSegment) = distance(l.𝘅₁.coord, l.𝘅₁.coord + l.𝘂)
