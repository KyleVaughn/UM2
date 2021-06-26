using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("Quadrilateral8_3D (N = $N)")
# Triangulation
for T in [Float32, Float64]
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 2)
    p₃ = Point_3D(T, 2, 3)
    p₄ = Point_3D(T, 0, 3)
    p₅ = Point_3D(T, 3//2, 1//2)
    p₆ = Point_3D(T, 5//2, 3//2)
    p₇ = Point_3D(T, 3//2, 5//2)
    p₈ = Point_3D(T, 0,    3//2)
    quad8 = [Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)) for i = 1:N]

    time = @belapsed triangulate.($quad8, 20)
    us_time = (time/1e-6)/N
    @printf("    Triangulation                  - %-9s: ", "$T")
    @printf("%10.2f μs\n", us_time) 
end

# Intersection
for T in [Float32, Float64]
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 2)
    p₃ = Point_3D(T, 2, 3)
    p₄ = Point_3D(T, 0, 3)
    p₅ = Point_3D(T, 3//2, 1//2)
    p₆ = Point_3D(T, 5//2, 3//2)
    p₇ = Point_3D(T, 3//2, 5//2)
    p₈ = Point_3D(T, 0,    3//2)
    quad8 = [Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)) for i = 1:N]
    l = LineSegment_3D(Point_3D(T, 1, 0, -2),
                       Point_3D(T, 1, 0,  2))

    @test !(l ∩ quad8[1])[1]
    time = @belapsed $l .∩ $quad8
    us_time = (time/1e-6)/N
    @printf("    0 Intersection (triangulation) - %-9s: ", "$T")
    @printf("%10.2f μs\n", us_time) 

    l = LineSegment_3D(Point_3D(T, 1, 1//2, -2),
                       Point_3D(T, 1, 1//2,  2))
    @test (l ∩ quad8[1])[1]
    @test (l ∩ quad8[1])[2] == 1
    @test (l ∩ quad8[1])[3] ≈ Point_3D(T, 1, 1//2)
    time = @belapsed $l .∩ $quad8
    us_time = (time/1e-6)/N
    @printf("    1 Intersection (triangulation) - %-9s: ", "$T")
    @printf("%10.2f μs\n", us_time) 

    p₂ = Point_3D(T, 2, 0, 3)
    quad8 = [Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)) for i = 1:N]
    l = LineSegment_3D(Point_3D(T, 0, 3, -2//10),
                       Point_3D(T, 3, 0, -2//10))
    @test (l ∩ quad8[1])[1]
    @test (l ∩ quad8[1])[2] == 2
    @test (l ∩ quad8[1])[3] ≈ Point_3D(T, 2.0432377, 0.9567623, -0.2)
    @test (l ∩ quad8[1])[4] ≈ Point_3D(T, 0.6839827, 2.3160174, -0.2)
    time = @belapsed $l .∩ $quad8
    us_time = (time/1e-6)/N
    @printf("    2 Intersection (triangulation) - %-9s: ", "$T")
    @printf("%10.2f μs\n", us_time) 
end

# Area
for T in [Float32, Float64]
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 2)
    p₃ = Point_3D(T, 2, 2)
    p₄ = Point_3D(T, 1, 1//4)
    p₅ = Point_3D(T, 3, 1)
    p₆ = Point_3D(T, 1, 1)
    p₇ = Point_3D(T, 3//2, 5//2)
    p₈ = Point_3D(T, 0,    3//2)
    quad8 = [Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈)) for i = 1:N]

    @test area(quad8[1]) ≈ 9.05095671164341
    time = @belapsed area.($quad8)
    us_time = (time/1e-6)/N
    @printf("    Area                           - %-9s: ", "$T")
    @printf("%10.2f μs\n", us_time) 
end
