using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Triangle_2D (N = $N)")

# Area
for type in [Float32, Float64]
    p₁ = Point_2D( type(0) )
    p₂ = Point_2D( type(1) )
    p₃ = Point_2D( type(1), type(1) )
    tri = [Triangle_2D((p₁, p₂, p₃)) for i = 1:N]

    time = @belapsed area.($tri)
    ns_time = (time/1e-9)/N
    @printf("    Area - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# In
for type in [Float32, Float64]
    p₁ = Point_2D( type(0) )
    p₂ = Point_2D( type(1) )
    p₃ = Point_2D( type(1), type(1) )
    tri = [Triangle_2D((p₁, p₂, p₃)) for i = 1:N]
    p = Point_2D( type(1//2), type(1//10))

    time = @belapsed $p .∈ $tri
    ns_time = (time/1e-9)/N
    @printf("    In - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end
