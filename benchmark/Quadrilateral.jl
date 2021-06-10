using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Quadrilateral (N = $N)")
# Intersection
for type in [Float32, Float64]
    p₁ = Point( type(0) )
    p₂ = Point( type(1) )
    p₃ = Point( type(1), type(1) )
    p₄ = Point( type(0), type(1) )
    quad = [Quadrilateral((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed triangulate.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Triangulation - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end


# Intersection
for type in [Float32, Float64]
    p₁ = Point( type(0) )
    p₂ = Point( type(1) )
    p₃ = Point( type(1), type(1) )
    p₄ = Point( type(0), type(1) )
    p₅ = Point(type.((0.9, 0.1, 5)))
    p₆ = Point(type.((0.9, 0.1, -5)))
    quad = [Quadrilateral((p₁, p₂, p₃, p₄)) for i = 1:N]
    l = [LineSegment(p₄, p₅) for i = 1:N] 

    time = @belapsed $l .∩ $quad
    ns_time = (time/1e-9)/N
    @printf("    Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# Area
for type in [Float32, Float64]
    p₁ = Point( type(0) )
    p₂ = Point( type(1) )
    p₃ = Point( type(1), type(1) )
    p₄ = Point( type(0), type(1) )
    quad = [Quadrilateral((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed area.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Area - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# In
for type in [Float32, Float64]
    p₁ = Point( type(0) )
    p₂ = Point( type(1) )
    p₃ = Point( type(1), type(1) )
    p₄ = Point( type(0), type(1) )
    quad = [Quadrilateral((p₁, p₂, p₃, p₄)) for i = 1:N]
    p = Point( type(1//2), type(1//10))

    time = @belapsed $p .∈ $quad
    ns_time = (time/1e-9)/N
    @printf("    In - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end
