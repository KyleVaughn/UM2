using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("Triangle_3D (N = $N)")

# Intersection
for T in [Float32, Float64]
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 1)
    p₃ = Point_3D(T, 1, 1)
    p₄ = Point_3D(T, 9//10, 1//10, -5)
    p₅ = Point_3D(T, 9//10, 1//10,  5)
    tri = [Triangle_3D((p₁, p₂, p₃)) for i = 1:N]
    l = LineSegment_3D(p₄, p₅)

    @test (l ∩ tri[1])[1]
    @test (l ∩ tri[1])[2] ≈ Point_3D{T}([0.9, 0.1, 0.0])
    time = @belapsed $l .∩ $tri
    ns_time = (time/1e-9)/N
    @printf("    Intersection                   - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end

# Area
for T in [Float32, Float64]
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 1)
    p₃ = Point_3D(T, 1, 1)
    tri = [Triangle_3D((p₁, p₂, p₃)) for i = 1:N]

    @test area(tri[1]) == 0.5
    time = @belapsed area.($tri)
    ns_time = (time/1e-9)/N
    @printf("    Area                           - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end
