using MOCNeutronTransport, BenchmarkTools, Test, Printf

N = Int(1E3)
println("QuadraticSegment (N = $N)")

# Intersection
for type in [Float32, Float64]
    x⃗₁ = Point( type.((0, 0, 0)) )
    x⃗₂ = Point( type.((2, 0, 0)) )
    x⃗₃ = Point( type.((1, 1, 0)) )
    x⃗₄ = Point( type.((0, 3, 0)) )
    x⃗₅ = Point( type.((2, 3, 0)) )
    l = [LineSegment(x⃗₄, x⃗₅) for i = 1:N]
    q = [QuadraticSegment(x⃗₁, x⃗₂, x⃗₃) for i = 1:N] 
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    0 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 

    x⃗₄ = Point( type.((1, 0, 0)) )
    x⃗₅ = Point( type.((1, 2, 0)) )
    q = [QuadraticSegment(x⃗₁, x⃗₂, x⃗₃) for i = 1:N]
    l = [LineSegment(x⃗₄, x⃗₅) for i = 1:N]
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    1 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 

    x⃗₄ = Point( type.((0, 0.75, 0)) )
    x⃗₅ = Point( type.((2, 0.75, 0)) )
    l = [LineSegment(x⃗₄, x⃗₅) for i = 1:N]
    time = @belapsed $l .∩ $q
    ns_time = (time/1e-9)/N
    @printf("    2 Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end
