using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Triangle_3D (N = $N)")

# Intersection
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(1) )
    p₃ = Point_3D( type(1), type(1) )
    p₄ = Point_3D(type.((0.9, 0.1, -5)))
    p₅ = Point_3D(type.((0.9, 0.1, 5)))
    tri = [Triangle_3D((p₁, p₂, p₃)) for i = 1:N]
    l = [LineSegment_3D(p₄, p₅) for i = 1:N] 

    time = @belapsed $l .∩ $tri
    ns_time = (time/1e-9)/N
    @printf("    Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# Area
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(1) )
    p₃ = Point_3D( type(1), type(1) )
    tri = [Triangle_3D((p₁, p₂, p₃)) for i = 1:N]

    time = @belapsed area.($tri)
    ns_time = (time/1e-9)/N
    @printf("    Area - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end
