using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("Triangle6_2D (N = $N)")

# Area
for T in [Float32, Float64]
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 2)
    p₃ = Point_2D(T, 2, 2)
    p₄ = Point_2D(T, 1, 1//4)
    p₅ = Point_2D(T, 3, 1)
    p₆ = Point_2D(T, 1, 1)
    tri = [Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    @test area(tri[1]) ≈ 3
    time = @belapsed area.($tri)
    ns_time = (time/1e-9)/N
    @printf("    Area                           - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)
end

# In
for T in [Float32, Float64]
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 2)
    p₃ = Point_2D(T, 2, 2)
    p₄ = Point_2D(T, 1, 1//4)
    p₅ = Point_2D(T, 3, 1)
    p₆ = Point_2D(T, 1, 1)
    tri = [Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]
    p = Point_2D(T, 1, 1//2)

    @test p ∈  tri[1]
    time = @belapsed $p .∈ $tri
    ns_time = (time/1e-9)/N
    @printf("    In                             - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)
end
