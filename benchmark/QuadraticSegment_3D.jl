using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("QuadraticSegment_3D (N = $N)")

# Intersection
for type in [Float32, Float64]
    x⃗₁ = Point_3D( type.((0, 0, 0)) )
    x⃗₂ = Point_3D( type.((2, 0, 0)) )
    x⃗₃ = Point_3D( type.((1, 1, 0)) )
    x⃗₄ = Point_3D( type.((0, 3, 0)) )
    x⃗₅ = Point_3D( type.((2, 3, 0)) )
    l = LineSegment_3D(x⃗₄, x⃗₅)
    q = [QuadraticSegment_3D(x⃗₁, x⃗₂, x⃗₃) for i = 1:N] 
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    0 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 

    x⃗₄ = Point_3D( type.((1, 0, 0)) )
    x⃗₅ = Point_3D( type.((1, 2, 0)) )
    q = [QuadraticSegment_3D(x⃗₁, x⃗₂, x⃗₃) for i = 1:N]
    l = [LineSegment_3D(x⃗₄, x⃗₅) for i = 1:N]
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    1 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 

    x⃗₄ = Point_3D( type.((0, 0.75, 0)) )
    x⃗₅ = Point_3D( type.((2, 0.75, 0)) )
    l = [LineSegment_3D(x⃗₄, x⃗₅) for i = 1:N]
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    2 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end
