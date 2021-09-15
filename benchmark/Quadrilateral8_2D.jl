using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("Quadrilateral8_2D (N = $N)")

# Area
for T in [Float32, Float64]
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 2)
    p₃ = Point_2D(T, 2, 3)
    p₄ = Point_2D(T, 0, 3)
    p₅ = Point_2D(T, 3//2, 1//2)
    p₆ = Point_2D(T, 5//2, 3//2)
    p₇ = Point_2D(T, 3//2, 5//2)
    p₈ = Point_2D(T, 0,    3//2)
    quad8 = [Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)) for i = 1:N]

    @test area(quad8[1]) ≈ 17//3
    time = @belapsed area.($quad8)
    ns_time = (time/1e-9)/N
    @printf("    Area                           - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)
end

# In
for T in [Float32, Float64]
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 2)
    p₃ = Point_2D(T, 2, 3)
    p₄ = Point_2D(T, 0, 3)
    p₅ = Point_2D(T, 3//2, 1//2)
    p₆ = Point_2D(T, 5//2, 3//2)
    p₇ = Point_2D(T, 3//2, 5//2)
    p₈ = Point_2D(T, 0,    3//2)
    quad8 = [Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)) for i = 1:N]
    p = Point_2D(T, 2, 3//2)

    @test p ∈  quad8[1]
    time = @belapsed $p .∈ $quad8
    ns_time = (time/1e-9)/N
    @printf("    In                             - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)
end

# Intersect
for T in [Float32, Float64]
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 2)
    p₃ = Point_2D(T, 2, 3)
    p₄ = Point_2D(T, 0, 3)
    p₅ = Point_2D(T, 3//2, 1//2)
    p₆ = Point_2D(T, 5//2, 3//2)
    p₇ = Point_2D(T, 3//2, 5//2)
    p₈ = Point_2D(T, 0, 3//2)
    quad8 = [Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)) for i = 1:N]

    l = LineSegment_2D(Point_2D(T, 0, -1), Point_2D(T, 4, -1))
    npoints, points = l ∩ quad8[1]
    @test npoints === 0
    time = @belapsed $l .∩ $quad8
    ns_time = (time/1e-9)/N
    @printf("    0 Intersection                 - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 

    l = LineSegment_2D(Point_2D(T, 3//2, -1), Point_2D(T, 3//2, 5))
    npoints, points = l ∩ quad8[1]
    @test npoints === 2
    @test points[1] ≈ Point_2D(T, 3//2, 1//2)
    @test points[2] ≈ Point_2D(T, 3//2, 5//2)
    time = @belapsed $l .∩ $quad8
    ns_time = (time/1e-9)/N
    @printf("    2 Intersection                 - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 

    l = LineSegment_2D(Point_2D(T, 0, 1//10), Point_2D(T, 4, 1//10))
    npoints, points = l ∩ quad8[1]
    @test npoints === 4
    @test points[1] ≈ Point_2D(T, 0.20557280900008415,       1//10)
    @test points[2] ≈ Point_2D(T, 1.9944271909999158,        1//10)
    @test points[3] ≈ Point_2D(T, 2.0644444444444447,        1//10)
    @test points[4] ≈ Point_2D(T, 0,                         1//10)
    time = @belapsed $l .∩ $quad8
    ns_time = (time/1e-9)/N
    @printf("    4 Intersection                 - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end
