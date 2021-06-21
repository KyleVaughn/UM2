using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Quadrilateral_3D (N = $N)")
# Triangulation
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(1) )
    p₃ = Point_3D( type(1), type(1) )
    p₄ = Point_3D( type(0), type(1) )
    quad = [Quadrilateral_3D((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed triangulate.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Triangulation - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end


# Intersection
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(1) )
    p₃ = Point_3D( type(1), type(1) )
    p₄ = Point_3D( type(0), type(1) )
    p₅ = Point_3D(type.((0.9, 0.1, 5)))
    p₆ = Point_3D(type.((0.9, 0.1, -5)))
    quad = [Quadrilateral_3D((p₁, p₂, p₃, p₄)) for i = 1:N]
    l = LineSegment_3D(p₄, p₅)

    time = @belapsed $l .∩ $quad
    ns_time = (time/1e-9)/N
    @printf("    Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# Area
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(1) )
    p₃ = Point_3D( type(1), type(1) )
    p₄ = Point_3D( type(0), type(1) )
    quad = [Quadrilateral_3D((p₁, p₂, p₃, p₄)) for i = 1:N]

    time = @belapsed area.($quad)
    ns_time = (time/1e-9)/N
    @printf("    Area - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end
