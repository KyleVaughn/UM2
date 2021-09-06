using CUDA
using MOCNeutronTransport
using BenchmarkTools
using Test

# Check num threads and give warning
nthreads = Threads.nthreads()
if nthreads === 1
    @warn "Only using single-thread for cpu. Try restarting julia with 'julia --threads n'"
else
    println("Using $nthreads threads for CPU multi-threading")
end

N = 2^20
println("Using Point arrays of length $N")

function sequential_add!(x, y, z)
    for i in eachindex(x, y)
        @inbounds z[i] = x[i] + y[i]
    end
    return nothing
end

function parallel_add!(x, y, z)
    Threads.@threads for i in eachindex(x, y)
        @inbounds z[i] = y[i] + x[i]
    end
    return nothing
end

function gpu_1_thread_add!(x, y, z)
    for i = 1:length(x)
        @inbounds z[i] = x[i] + y[i]
    end
    return nothing
end

function bench_gpu_1_thread_add!(x, y, z)
    CUDA.@sync begin
        @cuda gpu_1_thread_add!(x, y, z)
    end
end

function gpu_1_block_add!(x, y, z)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(x)
        @inbounds z[i] = x[i] + y[i]
    end
    return nothing
end

function bench_gpu_1_block_add!(x, y, z)
    CUDA.@sync begin
        @cuda threads=512 gpu_1_block_add!(x, y, z)
    end
end

function gpu_multiblock_add!(x, y, z)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds z[index] = x[index] + y[index]
    return nothing
end

function bench_gpu_multiblock_add!(x, y, z)
    numblocks = ceil(Int, length(x)/512)
    CUDA.@sync begin
        @cuda threads=512 blocks=numblocks gpu_multiblock_add!(x, y, z)
    end
end

cpu_time = 0.0
for T = [Float64, Float32]
    println("Using Point of type $T")
    x = fill(Point_2D(T, 1, 1), N)
    y = fill(Point_2D(T, 2, 2), N)
    z = fill(Point_2D(T, 0, 0), N)
    time = @belapsed sequential_add!($x, $y, $z)
    μs = 1e6*time
    if T == Float64 
        global cpu_time = μs
    end
    speedup = cpu_time/μs
    println("    CPU: single-thread = $μs μs")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(z .== Point_2D(T, 3, 3))

    
    fill!(z, Point_2D(T, 0, 0))
    time = @belapsed parallel_add!($x, $y, $z)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    CPU: $nthreads threads = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(z .== Point_2D(T, 3, 3))   
    
    x_d = CUDA.fill(Point_2D(T, 1, 1), N)
    y_d = CUDA.fill(Point_2D(T, 2, 2), N)
    z_d = CUDA.fill(Point_2D(T, 0, 0), N)
     
    time = @belapsed bench_gpu_1_thread_add!($x_d, $y_d, $z_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    GPU: single-thread/block, 1 blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(Array(z_d) .== Point_2D(T, 3, 3))
    
    fill!(z_d, Point_2D(T, 0, 0))
    time = @belapsed bench_gpu_1_block_add!($x_d, $y_d, $z_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    GPU: 512 threads/block, 1 blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(Array(z_d) .== Point_2D(T, 3, 3))
    
    fill!(z_d, Point_2D(T, 0, 0))
    time = @belapsed bench_gpu_multiblock_add!($x_d, $y_d, $z_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    numblocks = ceil(Int, N/512)
    println("    GPU: 512 threads/block, $numblocks blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(Array(z_d) .== Point_2D(T, 3, 3))
end
