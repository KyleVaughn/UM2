using CUDA
using MOCNeutronTransport
using BenchmarkTools
using Test

# Number of lines to use in vectors
N = 2^20
println("Using a LineSegment_2D vector of length $N")

# Check num threads and give warning
nthreads = Threads.nthreads()
if nthreads === 1
    @warn "Only using single-thread for cpu. Try restarting julia with 'julia --threads n'"
else
    println("Using $nthreads threads for CPU multi-threading")
end

# Single threads CPU intersection
function sequential_intersection!(out, l, lines)
    for i in eachindex(lines)
        @inbounds out[i] = l ∩ lines[i]
    end
    return nothing
end

# Multithreaded CPU intersection
function parallel_intersection!(out, l, lines)
    Threads.@threads for i in eachindex(lines)
        @inbounds out[i] = l ∩ lines[i]
    end
    return nothing
end

# Single threaded GPU intersection
function gpu_1_thread_intersection!(out, l, lines)
    for i = 1:length(lines)
        @inbounds out[i] = l ∩ lines[i]
    end
    return nothing
end

function bench_gpu_1_thread_intersection!(out, l, lines)
    CUDA.@sync begin
        @cuda gpu_1_thread_intersection!(out, l, lines)
    end
end

# Single block GPU intersection
function gpu_1_block_intersection!(out, l, lines)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(lines)
        @inbounds out[i] = l ∩ lines[i]
    end
    return nothing
end

function bench_gpu_1_block_intersection!(out, l, lines)
    CUDA.@sync begin
        @cuda threads=512 gpu_1_block_intersection!(out, l, lines)
    end
end

# Multiple blocks GPU intersection
function gpu_multiblock_intersection!(out, l, lines)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(lines)
        @inbounds out[i] = l ∩ lines[i]
    end
    return nothing
end

function bench_gpu_multiblock_intersection!(out, l, lines)
    numblocks = ceil(Int64, length(lines)/512)
    CUDA.@sync begin
        @cuda threads=512 blocks=numblocks gpu_multiblock_intersection!(out, l, lines)
    end
end

# Use the occupancy API to determine threads and blocks to saturate the GPU
function bench_gpu_multiblock_autooccupancy!(out, l, lines)
    kernel = @cuda launch=false gpu_multiblock_intersection!(out, l, lines)
    config = launch_configuration(kernel.fun)
    threads = min(length(lines), config.threads)
    blocks = cld(length(lines), threads)

    CUDA.@sync begin
        kernel(out, l, lines; threads, blocks)
    end
end

cpu_time = 0.0
for T = [Float64, Float32]
    println("Using LineSegment_2D of type $T")
    l = LineSegment_2D(Point_2D{T}(0, 0.2), Point_2D{T}(1, 0.8))
    lines = rand(LineSegment_2D{T}, N)
    out = fill((false, Point_2D{T}(0, 0)), N)
    ref = l .∩ lines
    time = @belapsed sequential_intersection!($out, $l, $lines)
    μs = 1e6*time
    if T == Float64 
        global cpu_time = μs
    end
    speedup = cpu_time/μs
    println("    CPU: single-thread = $μs μs")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    @test out == ref

    fill!(out, (false, Point_2D{T}(0, 0)))
    time = @belapsed parallel_intersection!($out, $l, $lines)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    CPU: $nthreads threads = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    @test out == ref
    
    lines_d = CuArray(lines)
    out_d = CUDA.fill((false, Point_2D{T}(0, 0)), N)
     
    time = @belapsed bench_gpu_1_thread_intersection!($out_d, $l, $lines_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    GPU: single-thread/block, 1 blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    out_arr = Array(out_d)
    @test getindex.(out_arr, 1) == getindex.(ref, 1)
    @test getindex.(out_arr, 2) ≈ getindex.(ref, 2)
    
    fill!(out_d, (false, Point_2D{T}(0, 0)))
    time = @belapsed bench_gpu_1_block_intersection!($out_d, $l, $lines_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    GPU: 512 threads/block, 1 blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    out_arr = Array(out_d)
    @test getindex.(out_arr, 1) == getindex.(ref, 1)
    @test getindex.(out_arr, 2) ≈ getindex.(ref, 2)

    fill!(out_d, (false, Point_2D{T}(0, 0)))
    time = @belapsed bench_gpu_multiblock_intersection!($out_d, $l, $lines_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    numblocks = ceil(Int64, N/512)
    println("    GPU: 512 threads/block, $numblocks blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    out_arr = Array(out_d)
    @test getindex.(out_arr, 1) == getindex.(ref, 1)
    @test getindex.(out_arr, 2) ≈ getindex.(ref, 2)

    fill!(out_d, (false, Point_2D{T}(0, 0)))
    time = @belapsed bench_gpu_multiblock_autooccupancy!($out_d, $l, $lines_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    kernel = @cuda launch=false gpu_multiblock_intersection!(out_d, l, lines_d)
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    println("    GPU: $threads threads/block, $blocks blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    out_arr = Array(out_d)
    @test getindex.(out_arr, 1) == getindex.(ref, 1)
    @test getindex.(out_arr, 2) ≈ getindex.(ref, 2)
end
