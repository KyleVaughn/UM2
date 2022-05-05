# Bounding box
# ---------------------------------------------------------------------------------------------
# Bounding box of a vector of points
function boundingbox(points::Vector{<:Point2D})
    xmin = ymin = typemax(T)
    xmax = ymax = typemin(T)
    for i = 1:length(points)
        x,y = points[i].coord  
        if x < xmin
            xmin = x
        end
        if xmax < x
            xmax = x
        end
        if y < ymin
            ymin = y
        end
        if ymax < y
            ymax = y
        end
    end
    return AABox2D(Point2D(xmin, ymin), 
                   Point2D(xmax, ymax))
end

function boundingbox(points::SVector{L, Point2D{T}}) where {L,T} 
    xmin = ymin = typemax(T)
    xmax = ymax = typemin(T)
    for i = 1:L
        x,y = points[i].coord  
        if x < xmin
            xmin = x
        end
        if xmax < x
            xmax = x
        end
        if y < ymin
            ymin = y
        end
        if ymax < y
            ymax = y
        end
    end
    return AABox2D(Point2D(xmin, ymin), 
                   Point2D(xmax, ymax))
end

# Bounding box of a vector of points
function boundingbox(points::Vector{<:Point3D})
    xmin = ymin = zmin = typemax(T)
    xmax = ymax = zmax = typemin(T)
    for i = 1:length(points)
        x,y,z = points[i].coord  
        if x < xmin
            xmin = x
        end
        if xmax < x
            xmax = x
        end
        if y < ymin
            ymin = y
        end
        if ymax < y
            ymax = y
        end
        if z < zmin
            zmin = z
        end
        if zmax < z
            zmax = z
        end
    end
    return AABox3D(Point3D(xmin, ymin, zmin), 
                   Point3D(xmax, ymax, zmax))
end

function boundingbox(points::SVector{L, Point3D{T}}) where {L,T} 
    xmin = ymin = zmin = typemax(T)
    xmax = ymax = zmax = typemin(T)
    for i = 1:L
        x,y,z = points[i].coord  
        if x < xmin
            xmin = x
        end
        if xmax < x
            xmax = x
        end
        if y < ymin
            ymin = y
        end
        if ymax < y
            ymax = y
        end
        if z < zmin
            zmin = z
        end
        if zmax < z
            zmax = z
        end
    end
    return AABox3D(Point3D(xmin, ymin, zmin), 
                   Point3D(xmax, ymax, zmax))
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

# Return the AABox which contains both bb₁ and bb₂
function Base.union(bb₁::AABox{Dim, T}, bb₂::AABox{Dim, T}) where {Dim, T}
    return AABox(Point{Dim, T}(min.(bb₁.minima.coord, bb₂.minima.coord)),
                 Point{Dim, T}(max.(bb₁.maxima.coord, bb₂.maxima.coord)))
end

# Bounding box
# ---------------------------------------------------------------------------------------------
boundingbox(poly::Polygon) = boundingbox(poly.points)

# Axis-aligned bounding box
function boundingbox(mesh::LinearUnstructuredMesh)
    # The bounding box may be determined entirely from the points.
    return boundingbox(mesh.points)
end

# Axis-aligned bounding box
function boundingbox(mesh::QuadraticUnstructuredMesh)
    return mapreduce(x->boundingbox(x), union, materialize_edges(mesh))
end
