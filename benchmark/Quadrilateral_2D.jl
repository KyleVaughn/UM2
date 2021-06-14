using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Quadrilateral_2D (N = $N)")
# Triangulation
for type in [Float32, Float64]
    p₁ = Point_2D( type(0) )
    p₂ = Point_2D( type(1) )
    p₃ = Point_2D( type(1), type(1) )
    p₄ = Point_2D( type(0), type(1) )
    quad = [Quadrilateral_2D((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed triangulate.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Triangulation - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# Area
for type in [Float32, Float64]
    p₁ = Point_2D( type(0) )
    p₂ = Point_2D( type(1) )
    p₃ = Point_2D( type(1), type(1) )
    p₄ = Point_2D( type(0), type(1) )
    quad = [Quadrilateral_2D((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed area.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Area - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# In
for type in [Float32, Float64]
    p₁ = Point_2D( type(0) )
    p₂ = Point_2D( type(1) )
    p₃ = Point_2D( type(1), type(1) )
    p₄ = Point_2D( type(0), type(1) )
    quad = [Quadrilateral_2D((p₁, p₂, p₃, p₄)) for i = 1:N]
    p = Point_2D( type(1//2), type(1//10))

    time = @belapsed $p .∈ $quad
    ns_time = (time/1e-9)/N
    @printf("    In - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end
