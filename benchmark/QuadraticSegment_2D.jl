using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("QuadraticSegment_2D (N = $N)")
for T in [Float32, Float64]
    x⃗₁ = Point_2D(T, 0, 0)
    x⃗₂ = Point_2D(T, 2, 0)
    x⃗₃ = Point_2D(T, 1, 1)
    q = [QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃) for i = 1:N]
    @test arc_length(q[1]) ≈ 2.9578856981569195
    time = @belapsed arc_length.($q)
    ns_time = (time/1e-9)/N
    @printf("    Arc length                     - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)
end

# Intersection
for T in [Float32, Float64]
    x⃗₁ = Point_2D(T, 0, 0)
    x⃗₂ = Point_2D(T, 2, 0)
    x⃗₃ = Point_2D(T, 1, 1)
    x⃗₄ = Point_2D(T, 0, 3)
    x⃗₅ = Point_2D(T, 2, 3)
    l = LineSegment_2D(x⃗₄, x⃗₅)
    q = [QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃) for i = 1:N]
    @test (l ∩ q[1])[1] == 0
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    0 Intersection                 - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)

    x⃗₄ = Point_2D(T, 1, 0)
    x⃗₅ = Point_2D(T, 1, 2)
    q = [QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃) for i = 1:N]
    l = LineSegment_2D(x⃗₄, x⃗₅)
    @test (l ∩ q[1])[1] == 1
    @test (l ∩ q[1])[2] == Point_2D{Float64}([1.0, 1.0])
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    1 Intersection                 - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)

    x⃗₄ = Point_2D(T, 0, 3//4)
    x⃗₅ = Point_2D(T, 2, 3//4)
    l = LineSegment_2D(x⃗₄, x⃗₅)
    @test (l ∩ q[1])[1] == 2
    @test (l ∩ q[1])[2] == Point_2D{Float64}([0.5, 0.75])
    @test (l ∩ q[1])[3] == Point_2D{Float64}([1.5, 0.75])
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    2 Intersection                 - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)
end
