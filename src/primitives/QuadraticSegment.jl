# A quadratic segment that passes through three points: 𝘅₁, 𝘅₂, and 𝘅₃.
#
# The segment satisfies:
# 𝗾(r) = (2r-1)(r-1)𝘅₁ + r(2r-1)𝘅₂ + 4r(1-r)𝘅₃, r ∈ [0,1]
# Equivalently, 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁, r ∈ [0,1] where:
# 𝘂 = 2(𝘅₁ + 𝘅₂ - 2𝘅₃) and 𝘃 = -(3𝘅₁ + 𝘅₂ - 4𝘅₃)
# The relation of the points may be seen in the diagram below.
#                 ___𝘅₃___
#            ____/        \____
#        ___/                  \
#     __/                       𝘅₂
#   _/
#  /
# 𝘅₁
#
# NOTE: 𝘅₃ is not necessarily the midpoint in real space, or even between 𝘅₁ and 𝘅₂, 
# but the curve starts at 𝘅₁, passes through 𝘅₃ at q(1/2), and ends at 𝘅₂.
struct QuadraticSegment{Dim, T} <:Edge{Dim, 2, T}
    points::SVector{3, Point{Dim, T}}
end

const QuadraticSegment2D = QuadraticSegment{2}
const QuadraticSegment3D = QuadraticSegment{3}

Base.@propagate_inbounds function Base.getindex(q::QuadraticSegment, i::Integer)
    getfield(q, :points)[i]
end
# Easily fetch 𝘂, 𝘃, in 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
# All branches but the correct one are pruned by the compiler, so this is fast
# when called inside a function.
function Base.getproperty(q::QuadraticSegment, sym::Symbol)
    if sym === :𝘂
        return 2(q[1] + q[2] - 2q[3])
    elseif sym === :𝘃
        return 4q[3] - 3q[1] - q[2]
    elseif sym === :𝘅₁
        return q[1] 
    elseif sym === :𝘅₂
        return q[2] 
    elseif sym === :𝘅₃
        return q[3] 
    else # fallback to getfield
        return getfield(q, sym)
    end
end

# Constructors
# ---------------------------------------------------------------------------------------------
function QuadraticSegment(p₁::Point{Dim, T}, 
                          p₂::Point{Dim, T}, 
                          p₃::Point{Dim, T}) where {Dim, T}
    return QuadraticSegment{Dim, T}(SVector{3, Point{Dim, T}}(p₁, p₂, p₃))
end
function QuadraticSegment{Dim}(p₁::Point{Dim, T}, 
                               p₂::Point{Dim, T}, 
                               p₃::Point{Dim, T}) where {Dim, T}
    return QuadraticSegment{Dim, T}(SVector{3, Point{Dim, T}}(p₁, p₂, p₃))
end

# Small methods
# ---------------------------------------------------------------------------------------------
# Interpolation
# Note: 𝗾(0) = 𝘅₁, 𝗾(1) = 𝘅₂, 𝗾(1/2) = 𝘅₃
(q::QuadraticSegment)(r) = Point(((2r-1)*(r-1))q.𝘅₁ + (r*(2r-1))q.𝘅₂ + (4r*(1-r))q.𝘅₃)
# Return the derivative of q, evalutated at r
# 𝗾′(r) = 2r𝘂 + 𝘃, which is simplified to below.
derivative(q::QuadraticSegment, r) = (4r - 3)*(q.𝘅₁ - q.𝘅₃) + (4r - 1)*(q.𝘅₂ - q.𝘅₃)
# Return the Jacobian of q, evalutated at r
jacobian(q::QuadraticSegment, r) = derivative(q, r) 

# Arc length
# ---------------------------------------------------------------------------------------------
# Return the arc length of the quadratic segment
#
# The arc length integral may be reduced to an integral over the square root of a 
# quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
#     1             1
# L = ∫ ‖𝗾′(r)‖dr = ∫ √(ar² + br + c) dr 
#     0             0
function arclength(q::QuadraticSegment)
    if isstraight(q)
        return distance(q.𝘅₁, q.𝘅₂)
    else
        𝘂 = q.𝘂
        𝘃 = q.𝘃
        a = 4(𝘂 ⋅ 𝘂)
        b = 4(𝘂 ⋅ 𝘃)
        c = 𝘃 ⋅ 𝘃
        # Compiler seems to catch the reused sqrt quantities for common subexpression
        # elimination, or computation is as quick as storage in a variable, so we 
        # leave the sqrts for readability
        l = ((2a + b)√(a + b + c) - b√c)/4a -
            (b^2 - 4a*c)/((2√a)^3)*log((2√a√(a + b + c) + (2a + b))/(2√a√c + b)) 
        return l 
    end
end

# Axis-aligned bounding box
# ---------------------------------------------------------------------------------------------
# Find the axis-aligned bounding box of the segment
#
# Find the extrema for x and y by finding the r_x such that dx/dr = 0 
# and r_y such that dy/dr = 0
# 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
# 𝗾′(r) = 2r𝘂 + 𝘃 ⟹  r_x, r_y = -𝘃 ./ 2𝘂
# Compare the extrema with the segment's endpoints to find the AABox
function boundingbox(q::QuadraticSegment{N}) where {N}
    𝘂 = q.𝘂
    𝘃 = q.𝘃
    𝗿 = 𝘃 ./ -2𝘂
    𝗽_stationary = 𝗿*𝗿*𝘂 + 𝗿*𝘃 + q.𝘅₁
    𝗽_min = min.(q.𝘅₁.coord, q.𝘅₂.coord)
    𝗽_max = max.(q.𝘅₁.coord, q.𝘅₂.coord)
    if N === 2
        xmin, ymin = 𝗽_min
        xmax, ymax = 𝗽_max
        if 0 < 𝗿[1] < 1
            xmin = min(𝗽_min[1], 𝗽_stationary[1])
            xmax = max(𝗽_max[1], 𝗽_stationary[1])
        end
        if 0 < 𝗿[2] < 1
            ymin = min(𝗽_min[2], 𝗽_stationary[2])
            ymax = max(𝗽_max[2], 𝗽_stationary[2])
        end
        return AABox2D(Point2D(xmin, ymin), Point2D(xmax, ymax))
    else # N === 3
        xmin, ymin, zmin = 𝗽_min
        xmax, ymax, zmax = 𝗽_max
        if 0 < 𝗿[1] < 1
            xmin = min(𝗽_min[1], 𝗽_stationary[1])
            xmax = max(𝗽_max[1], 𝗽_stationary[1])
        end
        if 0 < 𝗿[2] < 1
            ymin = min(𝗽_min[2], 𝗽_stationary[2])
            ymax = max(𝗽_max[2], 𝗽_stationary[2])
        end
        if 0 < 𝗿[3] < 1
            zmin = min(𝗽_min[3], 𝗽_stationary[3])
            zmax = max(𝗽_max[3], 𝗽_stationary[3])
        end
        return AABox3D(Point3D(xmin, ymin, zmin), Point3D(xmax, ymax, zmax))
    end
end

# Intersect
# ---------------------------------------------------------------------------------------------
# Intersection between a line segment and quadratic segment
#
# The quadratic segment: 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
# The line segment: 𝗹(s) = 𝘅₄ + s𝘄
# 𝘅₄ + s𝘄 = r²𝘂 + r𝘃 + 𝘅₁
# s𝘄 = r²𝘂 + r𝘃 + (𝘅₁ - 𝘅₄)
# 𝟬 = r²(𝘂 × 𝘄) + r(𝘃 × 𝘄) + (𝘅₁ - 𝘅₄) × 𝘄
# The cross product of two vectors in the plane is a vector of the form (0, 0, k).
# Let a = (𝘂 × 𝘄)ₖ, b = (𝘃 × 𝘄)ₖ, c = ([𝘅₁ - 𝘅₄] × 𝘄)ₖ
# 0 = ar² + br + c
# If a = 0 
#   r = -c/b
# else
#   r = (-b ± √(b²-4ac))/2a
# We must also solve for s
# 𝘅₄ + s𝘄 = 𝗾(r)
# s𝘄 = 𝗾(r) - 𝘅₄
# s = ([𝗾(r) - 𝘅₄] ⋅𝘄 )/(𝘄 ⋅ 𝘄)
#
# r is invalid if:
#   1) b² < 4ac
#   2) r ∉ [0, 1]   (Curve intersects, segment doesn't)
# s is invalid if:
#   1) s ∉ [0, 1]   (Line intersects, segment doesn't)
function intersect(l::LineSegment2D{T}, q::QuadraticSegment2D{T}) where {T}
    ϵ = T(5e-6) # Tolerance on r,s ∈ [-ϵ, 1 + ϵ]
    npoints = 0x0000
    p₁ = nan_point(Point2D{T})
    p₂ = nan_point(Point2D{T})
    if isstraight(q) # Use line segment intersection.
        # See LineSegment for the math behind this.
        𝘄 = q.𝘅₁ - l.𝘅₁
        𝘃 = q.𝘅₂ - q.𝘅₁
        z = l.𝘂 × 𝘃
        r = (𝘄 × 𝘃)/z
        s = (𝘄 × l.𝘂)/z
        if 1e-8 < abs(z) && -ϵ ≤ r ≤ 1 + ϵ && -ϵ ≤ s ≤ 1 + ϵ
            npoints += 0x0001
        end
        return npoints, SVector(l(r), p₂)
    else
        𝘂 = q.𝘂 
        𝘃 = q.𝘃 
        𝘄 = l.𝘂
        a = 𝘂 × 𝘄 
        b = 𝘃 × 𝘄
        c = (q.𝘅₁ - l.𝘅₁) × 𝘄
        w² = 𝘄 ⋅ 𝘄 
        if abs(a) < 1e-8
            r = -c/b
            -ϵ ≤ r ≤ 1 + ϵ || return 0x0000, SVector(p₁, p₂)
            p₁ = q(r)
            s = (p₁ - l.𝘅₁)⋅𝘄 
            if -ϵ*w² ≤ s ≤ (1 + ϵ)w²
                npoints = 0x0001
            end
        elseif b^2 ≥ 4a*c
            r₁ = (-b - √(b^2 - 4a*c))/2a
            r₂ = (-b + √(b^2 - 4a*c))/2a
            valid_p₁ = false
            if -ϵ ≤ r₁ ≤ 1 + ϵ
                p₁ = q(r₁)
                s₁ = (p₁ - l.𝘅₁)⋅𝘄
                if -ϵ*w² ≤ s₁ ≤ (1 + ϵ)w²
                    npoints += 0x0001
                    valid_p₁ = true
                end
            end
            if -ϵ ≤ r₂ ≤ 1 + ϵ
                p₂ = q(r₂)
                s₂ = (p₂ - l.𝘅₁)⋅𝘄
                if -ϵ*w² ≤ s₂ ≤ (1 + ϵ)w²
                    npoints += 0x0001
                end
            end
            if npoints === 0x0001 && !valid_p₁ 
                p₁ = p₂
            end
        end
        return npoints, SVector(p₁, p₂)
    end
end

# Intersect a line with a vector of quadratic edges
function intersect_edges(l::LineSegment{Dim, T}, edges::Vector{QuadraticSegment{Dim, T}}
                        ) where {Dim, T} 
    intersection_points = Point{Dim, T}[]
    for edge in edges 
        npoints, points = l ∩ edge 
        if 0 < hits
            append!(intersection_points, view(points, 1:hits))
        end
    end
    sort_intersection_points!(l, intersection_points)
    return intersection_points
end

# Intersect a vector of lines with a vector of quadratic edges
function intersect_edges(lines::Vector{LineSegment{Dim, T}}, 
                         edges::Vector{QuadraticSegment{Dim, T}}) where {Dim, T} 
    nlines = length(lines)
    intersection_points = [Point{Dim, T}[] for _ = 1:nlines]
    Threads.@threads for edge in edges 
        @inbounds for i = 1:nlines
            hits, points = lines[i] ∩ edge 
            if 0 < hits
                append!(intersection_points[i], view(points, 1:hits))
            end
        end
    end
    Threads.@threads for i = 1:nlines
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
    intersection_array_gpu = CUDA.fill(Point2D{T}(NaN, NaN), ceil(Int64, 2sqrt(nedges)), nlines)
    kernel = @cuda launch=false intersect_quadratic_edges_CUDA!(intersection_array_gpu, 
                                                                lines_gpu, edges_gpu)
    config = launch_configuration(kernel.fun)
    threads = min(nlines, config.threads)
    blocks = cld(nlines, threads)
    CUDA.@sync begin
        kernel(intersection_array_gpu, lines_gpu, edges_gpu; threads, blocks) 
    end 
    intersection_array = collect(intersection_array_gpu) 
    intersection_points = [ filter!(x->!isnan(x[1]), collect(intersection_array[:, i])) for i ∈ 1:nlines]
    Threads.@threads for i = 1:nlines
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
                for i = 1:hits
                    @inbounds intersection_points[ipt, index] = points[i]
                    ipt += 1
                end
            end                                                                 
        end
    end
    return nothing
end

# Is left 
# ---------------------------------------------------------------------------------------------
# If the point is left of the quadratic segment in the 2D plane. 
#   𝗽    ^
#   ^   /
# 𝘃 |  / 𝘂
#   | /
#   o
# If the segment is straight, or if the point is not within the bounding box of
# the segment, we can perform the isleft check with the straight line from the 
# segment's start point to the segment's end point.
# If these conditions aren't met, the segment's curvature must be accounted for.
# We find the point on the curve q that is nearest point to the point of interest. 
# Call this point q_near. We then perform the isleft check with the tangent vector 
# of q at q_near and the vector from q_near to p, p - q_near.
function isleft(p::Point2D, q::QuadraticSegment2D)
    if isstraight(q) || p ∉  boundingbox(q)
        𝘃₁ = q.𝘅₂ - q.𝘅₁
        𝘃₂ = p - q.𝘅₁
        return 𝘃₁ × 𝘃₂ > 0
    else
        # See nearest_point for an explanation of the math.
        # When the cubic has 3 real roots, the point must be inside the
        # curve of the segment. Meaning: 
        #   If the segment curves left, the point is right.
        #   If the segment curves right, the point is left.
        # This way we save substantial time by bypassing the complex number arithmetic
        𝘂 = q.𝘂
        𝘃 = q.𝘃
        𝘄 = p - q.𝘅₁
        # f′(r) = ar³ + br² + cr + d = 0
        a = 4(𝘂 ⋅ 𝘂)
        b = 6(𝘂 ⋅ 𝘃)
        c = 2((𝘃 ⋅ 𝘃) - 2(𝘂 ⋅𝘄))
        d = -2(𝘃 ⋅ 𝘄)
        # Lagrange's method
        e₁ = s₀ = -b/a
        e₂ = c/a
        e₃ = -d/a
        A = 2e₁^3 - 9e₁*e₂ + 27e₃
        B = e₁^2 - 3e₂
        if A^2 - 4B^3 > 0 # one real root
            s₁ = ∛((A + √(A^2 - 4B^3))/2)
            if s₁ == 0
                s₂ = s₁
            else
                s₂ = B/s₁
            end
            r = (s₀ + s₁ + s₂)/3
            𝘃₁ = 𝗗(q, r)
            𝘃₂ = p - q(r)
            return 𝘃₁ × 𝘃₂ > 0
        else # three real roots
            return (q.𝘅₂ - q.𝘅₁) × (q.𝘅₃ - q.𝘅₁) < 0
        end
    end
end

# Is straight
# ---------------------------------------------------------------------------------------------
# If the quadratic segment is effectively linear
#
# Check the sign of the cross product of the vectors (𝘅₃ - 𝘅₁) and (𝘅₂ - 𝘅₁)
# If the line is straight, 𝘅₃ - 𝘅₁ = c(𝘅₂ - 𝘅₁) where c ∈ (0, 1), hence
# (𝘅₃ - 𝘅₁) × (𝘅₂ - 𝘅₁) = 𝟬
function isstraight(q::QuadraticSegment2D)
    return abs((q.𝘅₃ - q.𝘅₁) × (q.𝘅₂ - q.𝘅₁)) < 1e-8
end
function isstraight(q::QuadraticSegment3D)
    return norm²((q.𝘅₃ - q.𝘅₁) × (q.𝘅₂ - q.𝘅₁)) < 1e-16
end

# Nearest point
# ---------------------------------------------------------------------------------------------
# Find the point on 𝗾(r) closest to the point of interest 𝘆. 
#
# Note: r ∈ [0, 1] is not necessarily true for this function, since it finds the minimizer
# of the function 𝗾(r), ∀r ∈ ℝ 
# Find r which minimizes ‖𝘆 - 𝗾(r)‖, where 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁. 
# This r also minimizes ‖𝘆 - 𝗾(r)‖²
# It can be shown that this is equivalent to finding the minimum of the quartic function
# ‖𝘆 - 𝗾(r)‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
# The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
# 𝘄 = 𝘆 - 𝘅₁, a = 4(𝘂 ⋅ 𝘂), b = 6(𝘂 ⋅ 𝘃), c = 2[(𝘃 ⋅ 𝘃) - 2(𝘂 ⋅𝘄)], d = -2(𝘃 ⋅ 𝘄)
# Lagrange's method (https://en.wikipedia.org/wiki/Cubic_equation#Lagrange's_method)
# is used to find the roots.
function nearest_point(p::Point{Dim,T}, q::QuadraticSegment) where {Dim,T}
    𝘂 = q.𝘂
    𝘃 = q.𝘃
    𝘄 = p - q.𝘅₁
    # f′(r) = ar³ + br² + cr + d = 0
    a = 4(𝘂 ⋅ 𝘂)
    b = 6(𝘂 ⋅ 𝘃)
    c = 2((𝘃 ⋅ 𝘃) - 2(𝘂 ⋅𝘄))   
    d = -2(𝘃 ⋅ 𝘄)
    # Lagrange's method
    e₁ = s₀ = -b/a
    e₂ = c/a
    e₃ = -d/a
    A = 2e₁^3 - 9e₁*e₂ + 27e₃
    B = e₁^2 - 3e₂
    if A^2 - 4B^3 > 0 # one real root
        s₁ = ∛((A + √(A^2 - 4B^3))/2)
        if s₁ == 0
            s₂ = s₁
        else
            s₂ = B/s₁
        end
        r = (s₀ + s₁ + s₂)/3
        return r, q(r)
    else # three real roots
        # Complex cube root
        t₁ = exp(log((A + √(complex(A^2 - 4B^3)))/2)/3)
        if t₁ == 0
            t₂ = t₁
        else
            t₂ = B/t₁
        end
        ζ₁ = Complex{T}(-1/2, √3/2)
        ζ₂ = conj(ζ₁)
        dist_min = typemax(T)
        r_near = zero(T)
        p_near = nan_point(Point{Dim,T})
        # Use the real part of each root
        for rᵢ in (real((s₀ +    t₁ +    t₂)/3), 
                   real((s₀ + ζ₂*t₁ + ζ₁*t₂)/3), 
                   real((s₀ + ζ₁*t₁ + ζ₂*t₂)/3))
            pᵢ = q(rᵢ)
            dist = distance²(pᵢ, p)
            if dist < dist_min
                dist_min = dist
                r_near = rᵢ
                p_near = pᵢ
            end
        end
        return r_near, p_near
    end 
end

# Random
# ---------------------------------------------------------------------------------------------
# Random quadratic segment in the Dim-dimensional unit hypercube
function Base.rand(::Type{QuadraticSegment{Dim, F}}) where {Dim, F} 
    points = rand(Point{Dim, F}, 3)
    return QuadraticSegment(points[1], points[2], points[3])
end

# N random quadratic segments in the Dim-dimensional unit hypercube
function Base.rand(::Type{QuadraticSegment{Dim, F}}, N::Int64) where {Dim, F}
    return [ rand(QuadraticSegment{Dim, F}) for i ∈ 1:N ]
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment)
        rr = LinRange(0, 1, 15)
        points = q.(rr)
        coords = reduce(vcat, [[points[i], points[i+1]] for i = 1:length(points)-1])
        return convert_arguments(LS, coords)
    end

    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:QuadraticSegment})
        point_sets = [convert_arguments(LS, q) for q in Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
    end
end
