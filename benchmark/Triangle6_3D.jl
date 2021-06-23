using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Triangle6_3D (N = $N)")
# Triangulation
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(2) )
    p₃ = Point_3D( type(2), type(2) )
    p₄ = Point_3D( type(1), type(1)/type(4) )
    p₅ = Point_3D( type(3), type(1) )
    p₆ = Point_3D( type(1), type(1) )
    tri = [Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    time = @belapsed triangulate.($tri, 13)
    us_time = (time/1e-6)/N
    @printf("    Triangulation - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 
end

# Intersection
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(2) )
    p₃ = Point_3D( type(2), type(2) )
    p₄ = Point_3D( type(1), type(1)/type(4) )
    p₅ = Point_3D( type(3), type(1) )
    p₆ = Point_3D( type(1), type(1) )
    tri = [Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    l = LineSegment_3D(Point_3D( type(1), type(0), type(-2)),
                       Point_3D( type(1), type(0), type(2)))
    time = @belapsed $l .∩ $tri
    us_time = (time/1e-6)/N
    @printf("    0 Intersection (triangulation) - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 

    l = LineSegment_3D(Point_3D( type(1), type(1)/type(2), type(-2)),
                       Point_3D( type(1), type(1)/type(2), type(2)))
    time = @belapsed $l .∩ $tri
    us_time = (time/1e-6)/N
    @printf("    1 Intersection (triangulation) - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 

    p₂ = Point_3D( type(2), type(0), type(3) )
    tri = [Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]
    l = LineSegment_3D(Point_3D( type(1), type(-2), type(0.2)),
                       Point_3D( type(1), type( 2), type(0.2)))
    time = @belapsed $l .∩ $tri
    us_time = (time/1e-6)/N
    @printf("    2 Intersection (triangulation) - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 
end

# intersect iteratively
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(2) )
    p₃ = Point_3D( type(2), type(2) )
    p₄ = Point_3D( type(1), type(1)/type(4) )
    p₅ = Point_3D( type(3), type(1) )
    p₆ = Point_3D( type(1), type(1) )
    tri = [Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    l = LineSegment_3D(Point_3D( type(1), type(0), type(-10)),
                       Point_3D( type(1), type(0), type(10)))
    time = @belapsed intersect_iterative.($l, $tri)
    us_time = (time/1e-6)/N
    @printf("    0 Intersection (iterative) - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 

    l = LineSegment_3D(Point_3D( type(1), type(1)/type(2), type(-10)),
                       Point_3D( type(1), type(1)/type(2), type(10)))
    time = @belapsed intersect_iterative.($l, $tri)
    us_time = (time/1e-6)/N
    @printf("    1 Intersection (iterative) - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 

    p₂ = Point_3D( type(2), type(0), type(3) )
    tri = [Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]
    l = LineSegment_3D(Point_3D( type(1), type(-10), type(0.2)),
                       Point_3D( type(1), type( 10), type(0.2)))
    time = @belapsed intersect_iterative.($l, $tri)
    us_time = (time/1e-6)/N
    @printf("    2 Intersection (iterative) - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 
end

# Area
for type in [Float32, Float64]
    p₁ = Point_3D( type(0) )
    p₂ = Point_3D( type(2) )
    p₃ = Point_3D( type(2), type(2) )
    p₄ = Point_3D( type(1), type(1)/type(4) )
    p₅ = Point_3D( type(3), type(1) )
    p₆ = Point_3D( type(1), type(1) )
    tri = [Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆)) for i = 1:N]

    time = @belapsed area.($tri)
    us_time = (time/1e-6)/N
    @printf("    Area - %-9s: ", "$type")
    @printf("%10.2f μs\n", us_time) 
end
