# real_to_parametric
p₁ = Point2D{T}(0, 0)
p₂ = Point2D{T}(2, 0)
p₃ = Point2D{T}(2, 2)
p₄ = Point2D{T}(3 // 2, 1 // 4)
p₅ = Point2D{T}(3, 1)
p₆ = Point2D{T}(1, 1)
tri6 = QuadraticTriangle(p₁, p₂, p₃, p₄, p₅, p₆)
@test tri6(real_to_parametric(p₁, tri6)) ≈ p₁
@test tri6(real_to_parametric(p₂, tri6)) ≈ p₂
@test tri6(real_to_parametric(p₃, tri6)) ≈ p₃
@test tri6(real_to_parametric(p₄, tri6)) ≈ p₄
@test tri6(real_to_parametric(p₅, tri6)) ≈ p₅
@test tri6(real_to_parametric(p₆, tri6)) ≈ p₆

# real_to_parametric
p₁ = Point2D{T}(0, 0)
p₂ = Point2D{T}(2, 0)
p₃ = Point2D{T}(2, 3)
p₄ = Point2D{T}(0, 3)
p₅ = Point2D{T}(3 // 2, 1 // 2)
p₆ = Point2D{T}(5 // 2, 3 // 2)
p₇ = Point2D{T}(3 // 2, 5 // 2)
p₈ = Point2D{T}(0, 3 // 2)
quad8 = QuadraticQuadrilateral(p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)
@test quad8(real_to_parametric(p₁, quad8)) ≈ p₁
@test quad8(real_to_parametric(p₂, quad8)) ≈ p₂
@test quad8(real_to_parametric(p₃, quad8)) ≈ p₃
@test quad8(real_to_parametric(p₄, quad8)) ≈ p₄
@test quad8(real_to_parametric(p₅, quad8)) ≈ p₅
@test quad8(real_to_parametric(p₆, quad8)) ≈ p₆
