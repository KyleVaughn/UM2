using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("QuadraticSegment_2D (N = $N)")

# Intersection
for type in [Float32, Float64]
    x⃗₁ = Point_2D( type.((0, 0)) )
    x⃗₂ = Point_2D( type.((2, 0)) )
    x⃗₃ = Point_2D( type.((1, 1)) )
    x⃗₄ = Point_2D( type.((0, 3)) )
    x⃗₅ = Point_2D( type.((2, 3)) )
    l = [LineSegment_2D(x⃗₄, x⃗₅) for i = 1:N]
    q = [QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃) for i = 1:N] 
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    0 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 

    x⃗₄ = Point_2D( type.((1, 0)) )
    x⃗₅ = Point_2D( type.((1, 2)) )
    q = [QuadraticSegment_2D(x⃗₁, x⃗₂, x⃗₃) for i = 1:N]
    l = [LineSegment_2D(x⃗₄, x⃗₅) for i = 1:N]
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    1 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 

    x⃗₄ = Point_2D( type.((0, 0.75)) )
    x⃗₅ = Point_2D( type.((2, 0.75)) )
    l = [LineSegment_2D(x⃗₄, x⃗₅) for i = 1:N]
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    2 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end