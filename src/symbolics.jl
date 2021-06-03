using Symbolics
using Latexify
using LinearAlgebra
@variables r s
@variables x₁ x₂ x₃ x₄ x₅ x₆ 
@variables y₁ y₂ y₃ y₄ y₅ y₆ 
@variables z₁ z₂ z₃ z₄ z₅ z₆ 
w₁ = (1 - r - s)*(2(1 - r - s) - 1)
w₂ = r*(2r-1)
w₃ = s*(2s-1)
w₄ = 4r*(1 - r - s)
w₅ = 4r*s
w₆ = 4s*(1 - r - s)
weights = [w₁, w₂, w₃, w₄, w₅, w₆]
p₁ = [x₁, y₁, z₁]
p₂ = [x₂, y₂, z₂]
p₃ = [x₃, y₃, z₃]
p₄ = [x₄, y₄, z₄]
p₅ = [x₅, y₅, z₅]
p₆ = [x₆, y₆, z₆]
points = [p₁, p₂, p₃, p₄, p₅, p₆]
d_dr = Differential(r)
d_ds = Differential(s)
tri6 = sum([w*p for (w, p) in zip(weights, points)])
dx_dr, dy_dr, dz_dr = [expand_derivatives(d_dr(tri6[i])) for i = 1:3]
dx_ds, dy_ds, dz_ds = [expand_derivatives(d_ds(tri6[i])) for i = 1:3]
tri6_dr = [dx_dr, dy_dr, dz_dr]
tri6_ds = [dx_ds, dy_ds, dz_ds]
tri6_dr × tri6_ds
