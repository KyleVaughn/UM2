# Intersect a line with a vector of quadratic edges
function intersect_edges(l::LineSegment{Dim, T},
                         edges::Vector{QuadraticSegment{Dim, T}}) where {Dim, T}
    intersection_points = Point{Dim, T}[]
    for edge in edges
        npoints, points = l ∩ edge
        0 < hits && append!(intersection_points, view(points, 1:hits))
    end
    sort_intersection_points!(l, intersection_points)
    return intersection_points
end

# Intersect a vector of lines with a vector of quadratic edges
function intersect_edges(lines::Vector{LineSegment{Dim, T}},
                         edges::Vector{QuadraticSegment{Dim, T}}) where {Dim, T}
    nlines = length(lines)
    intersection_points = [Point{Dim, T}[] for _ in 1:nlines]
    Threads.@threads for edge in edges
        @inbounds for i in 1:nlines
            hits, points = lines[i] ∩ edge
            0 < hits && append!(intersection_points[i], view(points, 1:hits))
        end
    end
    Threads.@threads for i in 1:nlines
        sort_intersection_points!(lines[i], intersection_points[i])
    end
    return intersection_points
end

function intersect_edges_CUDA(lines::Vector{LineSegment{2, T}},
                              edges::Vector{QuadraticSegment{2, T}}) where {T}
    nlines = length(lines)
    nedges = length(edges)
    # √(2*nedges) is a good guess for a square domain with even mesh distrubution, but what
    # about rectangular domains?
    lines_gpu = CuArray(lines)
    edges_gpu = CuArray(edges)
    intersection_array_gpu = CUDA.fill(nan(Point2D{T}), ceil(Int64, 2sqrt(nedges)), nlines)
    kernel = @cuda launch=false intersect_quadratic_edges_CUDA!(intersection_array_gpu,
                                                                lines_gpu, edges_gpu)
    config = launch_configuration(kernel.fun)
    threads = min(nlines, config.threads)
    blocks = cld(nlines, threads)
    CUDA.@sync begin kernel(intersection_array_gpu, lines_gpu, edges_gpu; threads, blocks) end
    intersection_array = collect(intersection_array_gpu)
    intersection_points = [filter!(x -> !isnan(x[1]), collect(intersection_array[:, i]))
                           for i in 1:nlines]
    Threads.@threads for i in 1:nlines
        sort_intersection_points!(lines[i], intersection_points[i])
    end
    return intersection_points
end

function intersect_quadratic_edges_CUDA!(intersection_points, lines, edges)
    nlines = length(lines)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    ipt = 1
    if index ≤ nlines
        for edge in edges
            hits, points = lines[index] ∩ edge
            if hits > 0
                for i in 1:hits
                    @inbounds intersection_points[ipt, index] = points[i]
                    ipt += 1
                end
            end
        end
    end
    return nothing
end
