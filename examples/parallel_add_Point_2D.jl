using CUDA
using MOCNeutronTransport
using BenchmarkTools
using Test

# Number of points to use in vectors
N = 2^20
println("Using Point arrays of length $N")

# Check num threads and give warning
nthreads = Threads.nthreads()
if nthreads === 1
    @warn "Only using single-thread for cpu. Try restarting julia with 'julia --threads n'"
else
    println("Using $nthreads threads for CPU multi-threading")
end

# Single threads CPU add
function sequential_add!(x, y, z)
    for i in eachindex(x, y)
        @inbounds z[i] = x[i] + y[i]
    end
    return nothing
end

# Multithreaded CPU add
function parallel_add!(x, y, z)
    Threads.@threads for i in eachindex(x, y)
        @inbounds z[i] = y[i] + x[i]
    end
    return nothing
end

# Single threaded GPU add
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

# Single block GPU add
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

# Multiple blocks GPU add
function gpu_multiblock_add!(x, y, z)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds z[index] = x[index] + y[index]
    end
    return nothing
end

function bench_gpu_multiblock_add!(x, y, z)
    numblocks = ceil(Int, length(x)/512)
    CUDA.@sync begin
        @cuda threads=512 blocks=numblocks gpu_multiblock_add!(x, y, z)
    end
end

# Use the occupancy API to determine threads and blocks to saturate the GPU
function bench_gpu_multiblock_autooccupancy!(x, y, z)
    kernel = @cuda launch=false gpu_multiblock_add!(x, y, z)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync begin
        kernel(x, y, z; threads, blocks)
    end
end

cpu_time = 0.0
for T = [Float64, Float32]
    println("Using Point of type $T")
    x = fill(Point_2D{T}(1, 1), N)
    y = fill(Point_2D{T}(2, 2), N)
    z = fill(Point_2D{T}(0, 0), N)
    time = @belapsed sequential_add!($x, $y, $z)
    μs = 1e6*time
    if T == Float64 
        global cpu_time = μs
    end
    speedup = cpu_time/μs
    println("    CPU: single-thread = $μs μs")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(z .== Point_2D{T}(3, 3))

    
    fill!(z, Point_2D{T}(0, 0))
    time = @belapsed parallel_add!($x, $y, $z)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    CPU: $nthreads threads = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(z .== Point_2D{T}(3, 3))   
    
    x_d = CUDA.fill(Point_2D{T}(1, 1), N)
    y_d = CUDA.fill(Point_2D{T}(2, 2), N)
    z_d = CUDA.fill(Point_2D{T}(0, 0), N)
     
    time = @belapsed bench_gpu_1_thread_add!($x_d, $y_d, $z_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    GPU: single-thread/block, 1 blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(Array(z_d) .== Point_2D{T}(3, 3))
    
    fill!(z_d, Point_2D{T}(0, 0))
    time = @belapsed bench_gpu_1_block_add!($x_d, $y_d, $z_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    GPU: 512 threads/block, 1 blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(Array(z_d) .== Point_2D{T}(3, 3))
    
    fill!(z_d, Point_2D{T}(0, 0))
    time = @belapsed bench_gpu_multiblock_add!($x_d, $y_d, $z_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    numblocks = ceil(Int64, N/512)
    println("    GPU: 512 threads/block, $numblocks blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(Array(z_d) .== Point_2D{T}(3, 3))

    fill!(z_d, Point_2D{T}(0, 0))
    time = @belapsed bench_gpu_multiblock_autooccupancy!($x_d, $y_d, $z_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    kernel = @cuda launch=false gpu_multiblock_add!(x_d, y_d, z_d)
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    println("    GPU: $threads threads/block, $blocks blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test all(Array(z_d) .== Point_2D{T}(3, 3))
end

