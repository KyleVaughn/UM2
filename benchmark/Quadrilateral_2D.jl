using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Quadrilateral_2D (N = $N)")
# Triangulation
for T in [Float32, Float64]
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 1)
    p₃ = Point_2D(T, 1, 1)
    p₄ = Point_2D(T, 0, 1)
    quad = [Quadrilateral_2D((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed triangulate.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Triangulation - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end

# Area
for T in [Float32, Float64]
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 1)
    p₃ = Point_2D(T, 1, 1)
    p₄ = Point_2D(T, 0, 1)
    quad = [Quadrilateral_2D((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed area.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Area - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end

# In
for T in [Float32, Float64]
    p₁ = Point_2D(T, 0)
    p₂ = Point_2D(T, 1)
    p₃ = Point_2D(T, 1, 1)
    p₄ = Point_2D(T, 0, 1)
    quad = [Quadrilateral_2D((p₁, p₂, p₃, p₄)) for i = 1:N]
    p = Point_2D(T, 1//2, 1//10)

    time = @belapsed $p .∈ $quad
    ns_time = (time/1e-9)/N
    @printf("    In - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end
