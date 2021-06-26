using MOCNeutronTransport, BenchmarkTools, Printf, Test

N = Int(1E3)
println("Point_3D (N = $N)")

# Addition
for T in [Float32, Float64]
    p1 = [Point_3D(T, 1, 2, 0) for i = 1:N] 
    p2 = [Point_3D(T, 2, 4, 6) for i = 1:N]

    # Point_3D addition
    @test all((p1 .+ p2) .== Point_3D(T, 3, 6, 6))
    time = @belapsed $p1 .+ $p2
    ns_time = (time/1e-9)/N
    @printf("    Addition                       - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end

# Cross product
for T in [Float32, Float64]
    p1 = [Point_3D(T, 1, 2, 0) for i = 1:N] 
    p2 = [Point_3D(T, 2, 4, 6) for i = 1:N]
    # Cross product
    @test all((p1 .× p2) .== Point_3D(T, 12, -6, 0))
    time = @belapsed $p1 .× $p2
    ns_time = (time/1e-9)/N
    @printf("    Cross product                  - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end
