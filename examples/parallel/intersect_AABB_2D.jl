using CUDA
using MOCNeutronTransport
using BenchmarkTools
using Test
using StaticArrays

# Number of rects to use in vectors
N = 2^20
println("Using a AABB_2D vector of length $N")

# Check num threads and give warning
nthreads = Threads.nthreads()
if nthreads === 1
    @warn "Only using single-thread for cpu. Try restarting julia with 'julia --threads n'"
else
    println("Using $nthreads threads for CPU multi-threading")
end

# Single threads CPU intersection
function sequential_intersection!(out, l, rects)
    for i in eachindex(rects)
        @inbounds out[i] = l ∩ rects[i]
    end
    return nothing
end

# Multithreaded CPU intersection
function parallel_intersection!(out, l, rects)
    Threads.@threads for i in eachindex(rects)
        @inbounds out[i] = l ∩ rects[i]
    end
    return nothing
end

# Single threaded GPU intersection
function gpu_1_thread_intersection!(out, l, rects)
    for i = 1:length(rects)
        @inbounds out[i] = l ∩ rects[i]
    end
    return nothing
end

function bench_gpu_1_thread_intersection!(out, l, rects)
    CUDA.@sync begin
        @cuda gpu_1_thread_intersection!(out, l, rects)
    end
end

# Single block GPU intersection
function gpu_1_block_intersection!(out, l, rects)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(rects)
        @inbounds out[i] = l ∩ rects[i]
    end
    return nothing
end

function bench_gpu_1_block_intersection!(out, l, rects)
    CUDA.@sync begin
        @cuda threads=512 gpu_1_block_intersection!(out, l, rects)
    end
end

# Multiple blocks GPU intersection
function gpu_multiblock_intersection!(out, l, rects)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(rects)
        @inbounds out[i] = l ∩ rects[i]
    end
    return nothing
end

function bench_gpu_multiblock_intersection!(out, l, rects)
    numblocks = ceil(Int64, length(rects)/512)
    CUDA.@sync begin
        @cuda threads=512 blocks=numblocks gpu_multiblock_intersection!(out, l, rects)
    end
end

# Use the occupancy API to determine threads and blocks to saturate the GPU
function bench_gpu_multiblock_autooccupancy!(out, l, rects)
    kernel = @cuda launch=false gpu_multiblock_intersection!(out, l, rects)
    config = launch_configuration(kernel.fun)
    threads = min(length(rects), config.threads)
    blocks = cld(length(rects), threads)

    CUDA.@sync begin
        kernel(out, l, rects; threads, blocks)
    end
end

cpu_time = 0.0
for T = [Float64, Float32]
    println("Using AABB_2D of type $T")
    l = LineSegment_2D(Point_2D{T}(0, 0.2), Point_2D{T}(1, 0.8))
    rects = rand(AABB_2D{T}, N)
    out = fill((false, SVector(Point_2D{T}(0, 0), Point_2D{T}(0, 0))), N)
    ref = l .∩ rects
    time = @belapsed sequential_intersection!($out, $l, $rects)
    μs = 1e6*time
    if T == Float64 
        global cpu_time = μs
    end
    speedup = cpu_time/μs
    println("    CPU: single-thread = $μs μs")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    @test out == ref

    fill!(out, (false, SVector(Point_2D{T}(0, 0), Point_2D{T}(0, 0))))
    time = @belapsed parallel_intersection!($out, $l, $rects)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    CPU: $nthreads threads = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    @test out == ref
    
    rects_d = CuArray(rects)
    out_d = CUDA.fill((false, SVector(Point_2D{T}(0, 0), Point_2D{T}(0, 0))), N)
     
    time = @belapsed bench_gpu_1_thread_intersection!($out_d, $l, $rects_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    GPU: single-thread/block, 1 blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    @test Array(out_d) == ref
    
    fill!(out_d, (false, SVector(Point_2D{T}(0, 0), Point_2D{T}(0, 0))))
    time = @belapsed bench_gpu_1_block_intersection!($out_d, $l, $rects_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    println("    GPU: 512 threads/block, 1 blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    @test Array(out_d) == ref

    fill!(out_d, (false, SVector(Point_2D{T}(0, 0), Point_2D{T}(0, 0))))
    time = @belapsed bench_gpu_multiblock_intersection!($out_d, $l, $rects_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    numblocks = ceil(Int64, N/512)
    println("    GPU: 512 threads/block, $numblocks blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 = $speedup")
    @test Array(out_d) == ref

    fill!(out_d, (false, SVector(Point_2D{T}(0, 0), Point_2D{T}(0, 0))))
    time = @belapsed bench_gpu_multiblock_autooccupancy!($out_d, $l, $rects_d)
    μs = 1e6*time
    speedup = cpu_time/μs
    kernel = @cuda launch=false gpu_multiblock_intersection!(out_d, l, rects_d)
    config = launch_configuration(kernel.fun)
    threads = min(N, config.threads)
    blocks = cld(N, threads)
    println("    GPU: $threads threads/block, $blocks blocks = $μs μs.")
    println("         Speed up compared to single-thread CPU & Float64 Points = $speedup")
    @test Array(out_d) == ref
end
