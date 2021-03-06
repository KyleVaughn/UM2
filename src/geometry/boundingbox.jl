export boundingbox

function boundingbox(points::Vector{Point{2, T}}) where {T}
    xmin = ymin = typemax(T)
    xmax = ymax = typemin(T)
    for i in 1:length(points)
        x, y = points[i]
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        xmax = max(xmax, x) 
        ymax = max(ymax, y) 
    end
    return AABox{2, T}(Point{2, T}(xmin, ymin), Point{2, T}(xmax, ymax))
end

function boundingbox(points::Vec{L, Point{2, T}}) where {L, T}
    xmin = ymin = typemax(T)
    xmax = ymax = typemin(T)
    for i in 1:L
        x, y = points[i]
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        xmax = max(xmax, x) 
        ymax = max(ymax, y) 
    end
    return AABox{2, T}(Point{2, T}(xmin, ymin), Point{2, T}(xmax, ymax))
end

function boundingbox(l::LineSegment{Point{2, T}}) where {T}
    return boundingbox(l.vertices)::AABox{2, T}
end

function boundingbox(q::QuadraticSegment{Point{2, T}}) where {T}
    # Find the extrema for x and y by finding the r_x such that dx/dr = 0    
    # and r_y such that dy/dr = 0    
    # q(r) = Pâ + rð + rÂ²ð
    # qâ²(r) = ð + 2rð â¹  r_x, r_y = -ð â 2ð    
    # Compare the extrema with the segment's endpoints to find the AABox    
    q1 = coordinates(q[1])
    q2 = coordinates(q[2])
    q3 = coordinates(q[3])
    ðââ = q3 - q1
    ðââ = q3 - q2
    ð = 3ðââ + ðââ
    ð = -2(ðââ + ðââ)
    ð¿ = ð â -2ð  
    P_stationary = @. q1 + ð¿ * ð + ð¿ * ð¿ * ð
    P_min = min.(q1, q2)       
    P_max = max.(q1, q2)       
    xmin, ymin = P_min                     
    xmax, ymax = P_max                     
    if 0 < ð¿[1] < 1                        
        xmin = min(P_min[1], P_stationary[1])    
        xmax = max(P_max[1], P_stationary[1])    
    end    
    if 0 < ð¿[2] < 1    
        ymin = min(P_min[2], P_stationary[2])    
        ymax = max(P_max[2], P_stationary[2])    
    end    
    return AABox{2, T}(Point{2, T}(xmin, ymin), Point{2, T}(xmax, ymax))    
end

function boundingbox(poly::Polygon{N, Point{D, T}}) where {N, D, T}
    return boundingbox(vertices(poly))::AABox{D, T}
end

function boundingbox(poly::QuadraticPolygon{N, Point{D, T}})::AABox{D, T} where {N, D, T}
    return mapreduce(boundingbox, âª, edges(poly))
end
