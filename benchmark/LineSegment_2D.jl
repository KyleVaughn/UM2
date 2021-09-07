using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("LineSegment_2D (N = $N)")

# Intersection
for T in [Float32, Float64]
    l1 =  LineSegment_2D(Point_2D(T, 0,  1), Point_2D(T, 2, -1))
    l2 = [LineSegment_2D(Point_2D(T, 0, -1), Point_2D(T, 2,  1)) for i = 1:N]
    @test (l1 ∩ l2[1])[1] == 1
    @test (l1 ∩ l2[1])[2] == Point_2D{Float64}([1.0, 0.0])
    time = @belapsed $l1 .∩ $l2
    ns_time = (time/1e-9)/N
    @printf("    Intersection                   - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end
