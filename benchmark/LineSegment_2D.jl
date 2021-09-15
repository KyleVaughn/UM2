using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("LineSegment_2D (N = $N)")
# Arc length
for T in [Float32, Float64]
    p₁ = Point_2D(T, 1, 2)
    p₂ = Point_2D(T, 2, 4)
    l = [LineSegment_2D(p₁, p₂) for i = 1:N]
    @test arc_length(l[1]) ≈ sqrt(T(5))
    time = @belapsed arc_length.($l)
    ns_time = (time/1e-9)/N
    @printf("    Arc length                     - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time)
end

# Intersection
for T in [Float32, Float64]
    l1 =  LineSegment_2D(Point_2D(T, 0,  0), Point_2D(T, 2, 0))
    l2 = [LineSegment_2D(Point_2D(T, 1,  2), Point_2D(T, 1, 1//10)) for i = 1:N]
    @test (l1 ∩ l2[1])[1] === 0
    time = @belapsed $l1 .∩ $l2
    ns_time = (time/1e-9)/N
    @printf("    0 Intersection                 - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 


    l1 =  LineSegment_2D(Point_2D(T, 0,  1), Point_2D(T, 2, -1))
    l2 = [LineSegment_2D(Point_2D(T, 0, -1), Point_2D(T, 2,  1)) for i = 1:N]
    @test (l1 ∩ l2[1])[1] == 1
    @test (l1 ∩ l2[1])[2][1] ≈ Point_2D(T, 1, 0)
    time = @belapsed $l1 .∩ $l2
    ns_time = (time/1e-9)/N
    @printf("    1 Intersection                 - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end
