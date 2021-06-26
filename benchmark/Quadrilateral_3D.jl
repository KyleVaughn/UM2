using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("Quadrilateral_3D (N = $N)")
# Triangulation
for T in [Float32, Float64]
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 1)
    p₃ = Point_3D(T, 1, 1)
    p₄ = Point_3D(T, 0, 1)
    quad = [Quadrilateral_3D((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed triangulate.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Triangulation                  - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end


# Intersection
for T in [Float32, Float64]
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 1)
    p₃ = Point_3D(T, 1, 1)
    p₄ = Point_3D(T, 0, 1)
    p₅ = Point_3D(T, 9//10, 1//10,  5)
    p₆ = Point_3D(T, 9//10, 1//10, -5)
    quad = [Quadrilateral_3D((p₁, p₂, p₃, p₄)) for i = 1:N]
    l = LineSegment_3D(p₄, p₅)

    @test (l ∩ quad[1])[1]
    @test (l ∩ quad[1])[2] ≈ Point_3D(T, 0, 1, 0)
    time = @belapsed $l .∩ $quad
    ns_time = (time/1e-9)/N
    @printf("    Intersection                   - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end

# Area
for T in [Float32, Float64]
    p₁ = Point_3D(T, 0)
    p₂ = Point_3D(T, 1)
    p₃ = Point_3D(T, 1, 1)
    p₄ = Point_3D(T, 0, 1)
    quad = [Quadrilateral_3D((p₁, p₂, p₃, p₄)) for i = 1:N]

    @test area(quad[1]) ≈ 1 
    time = @belapsed area.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Area                           - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end
