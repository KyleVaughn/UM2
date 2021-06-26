using MOCNeutronTransport, BenchmarkTools, Printf

N = Int(1E3)
println("Point_2D (N = $N)")

# Addition
for T in [Float32, Float64]
    p1 = [Point_2D(T, 1, 2) for i = 1:N] 
    p2 = [Point_2D(T, 2, 4) for i = 1:N]

    # Point_2D addition
    time = @belapsed $p1 .+ $p2
    ns_time = (time/1e-9)/N
    @printf("    Addition     - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end

# Cross product
for T in [Float32, Float64]
    p1 = [Point_2D(T, 1, 2) for i = 1:N] 
    p2 = [Point_2D(T, 2, 4) for i = 1:N]
    # Cross product
    time = @belapsed $p1 .× $p2
    ns_time = (time/1e-9)/N
    @printf("    Cross product - %-9s: ", "$T")
    @printf("%10.2f ns\n", ns_time) 
end
