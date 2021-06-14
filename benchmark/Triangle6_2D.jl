using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Triangle6_2D (N = $N)")
# Triangulation
for type in [Float32, Float64]
    p₁ = Point_2D( type(0) )
    p₂ = Point_2D( type(2) )
    p₃ = Point_2D( type(2), type(2) )
    p₄ = Point_2D( type(1), type(1)/type(4) )
    p₅ = Point_2D( type(3), type(1) )
    p₆ = Point_2D( type(1), type(1) )
    tri = [Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    time = @belapsed triangulate.($tri, 13)
    us_time = (time/1e-6)/N
    @printf("    Triangulation - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 
end

# Area
for type in [Float32, Float64]
    p₁ = Point_2D( type(0) )
    p₂ = Point_2D( type(2) )
    p₃ = Point_2D( type(2), type(2) )
    p₄ = Point_2D( type(1), type(1)/type(4) )
    p₅ = Point_2D( type(3), type(1) )
    p₆ = Point_2D( type(1), type(1) )
    tri = [Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    time = @belapsed area.($tri)
    ns_time = (time/1e-9)/N
    @printf("    Area - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# In
for type in [Float32, Float64]
    p₁ = Point_2D( type(0) )
    p₂ = Point_2D( type(2) )
    p₃ = Point_2D( type(2), type(2) )
    p₄ = Point_2D( type(1), type(1)/type(4) )
    p₅ = Point_2D( type(3), type(1) )
    p₆ = Point_2D( type(1), type(1) )
    tri = [Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]
    p = Point_2D( type(1), type(1//2))

    time = @belapsed $p .∈ $tri
    us_time = (time/1e-6)/N
    @printf("    In - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 
end
