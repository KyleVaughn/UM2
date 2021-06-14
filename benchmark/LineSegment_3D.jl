using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("LineSegment_3D (N = $N)")

# Intersection
for type in [Float32, Float64]
    l1 = [LineSegment_3D(Point_3D( type.((0, 1)) ), Point_3D( type.((2, -1)) )) for i = 1:N]
    l2 = [LineSegment_3D(Point_3D( type.((0, -1)) ), Point_3D( type.((2, 1)) )) for i = 1:N]
    time = @belapsed $l1 .∩ $l2
    ns_time = (time/1e-9)/N
    @printf("    Intersection - %-9s: ", "$type")
    @printf("%10.2f ns\n", ns_time) 
end
