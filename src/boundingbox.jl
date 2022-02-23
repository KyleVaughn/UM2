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

