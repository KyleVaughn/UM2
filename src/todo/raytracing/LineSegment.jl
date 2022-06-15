# Intersect a vector of lines with a vector of linear edges
function intersect_edges(lines::Vector{LineSegment{Dim, T}}, 
                         edges::Vector{LineSegment{Dim, T}}) where {Dim, T} 
    nlines = length(lines)
    intersection_points = [Point{Dim, T}[] for _ = 1:nlines]
    Threads.@threads for edge in edges 
        @inbounds for i = 1:nlines
            hit, point = lines[i] ∩ edge 
            hit && push!(intersection_points[i], point)
        end
    end
    Threads.@threads for i = 1:nlines
        sort_intersection_points!(lines[i], intersection_points[i])
    end
    return intersection_points
end

# Intersect a vector of lines with a vector of linear edges, using CUDA
function intersect_edges_CUDA(lines::Vector{LineSegment{2, T}}, 
                              edges::Vector{LineSegment{2, T}}) where {T} 
    nlines = length(lines)
    nedges = length(edges)
    # √(2*nedges) is a good guess for a square domain with even mesh distrubution, but what
    # about rectangular domains?
    lines_gpu = CuArray(lines)
    edges_gpu = CuArray(edges)
    intersection_array_gpu = CUDA.fill(Point2D{T}(NaN, NaN), ceil(Int64, 2sqrt(nedges)), nlines)
    kernel = @cuda launch=false _intersect_linear_edges_CUDA!(intersection_array_gpu, 
                                                              lines_gpu, edges_gpu)
    config = launch_configuration(kernel.fun)
    threads = min(nlines, config.threads)
    blocks = cld(nlines, threads)
    CUDA.@sync begin
        kernel(intersection_array_gpu, lines_gpu, edges_gpu; threads, blocks) 
    end 
    intersection_array = collect(intersection_array_gpu) 
    intersection_points = [ filter!(x->!isnan(x[1]), 
                                    collect(intersection_array[:, i])) for i ∈ 1:nlines]
    Threads.@threads for i = 1:nlines
        sort_intersection_points!(lines[i], intersection_points[i])
    end
    return intersection_points
end

function _intersect_linear_edges_CUDA!(intersection_points, lines, edges)
    nlines = length(lines)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    ipt = 1
    if index ≤ nlines
        for edge in edges 
            hit, point = lines[index] ∩ edge
            if hit
                @inbounds intersection_points[ipt, index] = point
                ipt += 1
            end                                                                 
        end
    end
    return nothing
end


# Sort
# ---------------------------------------------------------------------------------------------
# Sort intersection points along a line segment, deleting points that are less than 
# the minimum_segment_length apart
function sort_intersection_points!(l::LineSegment, points::Vector{<:Point})
    sort!(l.𝘅₁, points)
    id_start = 1 
    n = length(points)
    deletion_indices = Int64[]
    for id_stop ∈ 2:n
        if distance²(points[id_start], points[id_stop]) < minimum_segment_length^2
            push!(deletion_indices, id_stop)
        else
            id_start = id_stop
        end
    end
    deleteat!(points, deletion_indices)
    return points
end

# Find the face containing the point p
function findface(p::Point, faces::Vector{<:Face})
    @inbounds for i ∈ 1:length(faces)
        p ∈ faces[i] && return i
    end
    return 0
end
