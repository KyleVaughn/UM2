using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Triangle6 (N = $N)")

# Intersection
for type in [Float32, Float64]
    p₁ = Point( type(0) )
    p₂ = Point( type(2) )
    p₃ = Point( type(2), type(2) )
    p₄ = Point( type(1), type(1)/type(4) )
    p₅ = Point( type(3), type(1) )
    p₆ = Point( type(1), type(1) )
    tri = [Triangle6((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    l = [LineSegment(Point( type(1), type(0), type(-2)),
                     Point( type(1), type(0), type(2))) for i = 1:N]
    time = @belapsed $l .∩ $tri
    us_time = (time/1e-6)/N
    @printf("    0 Intersection - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 

    l = [LineSegment(Point( type(1), type(1)/type(2), type(-2)),
                     Point( type(1), type(1)/type(2), type(2))) for i = 1:N]
    time = @belapsed $l .∩ $tri
    us_time = (time/1e-6)/N
    @printf("    1 Intersection - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 

    p₂ = Point( type(2), type(0), type(3) )
    tri = [Triangle6((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]
    l = [LineSegment(Point( type(1), type(-2), type(0.2)),
                     Point( type(1), type( 2), type(0.2))) for i = 1:N]
    time = @belapsed $l .∩ $tri
    us_time = (time/1e-6)/N
    @printf("    2 Intersection - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 
end

# Area
for type in [Float32, Float64]
    p₁ = Point( type(0) )
    p₂ = Point( type(2) )
    p₃ = Point( type(2), type(2) )
    p₄ = Point( type(1), type(1)/type(4) )
    p₅ = Point( type(3), type(1) )
    p₆ = Point( type(1), type(1) )
    tri = [Triangle6((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    time = @belapsed area.($tri)
    ns_time = (time/1e-9)/N
    @printf("    Area - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end

# In
for type in [Float32, Float64]
    p₁ = Point( type(0) )
    p₂ = Point( type(2) )
    p₃ = Point( type(2), type(2) )
    p₄ = Point( type(1), type(1)/type(4) )
    p₅ = Point( type(3), type(1) )
    p₆ = Point( type(1), type(1) )
    tri = [Triangle6((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]
    p = Point( type(1), type(1//2))

    time = @belapsed $p .∈ $tri
    us_time = (time/1e-6)/N
    @printf("    In - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 
end
