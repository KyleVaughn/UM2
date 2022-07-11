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
    # q(r) = P₁ + r𝘂 + r²𝘃
    # q′(r) = 𝘂 + 2r𝘃 ⟹  r_x, r_y = -𝘂 ⊘ 2𝘃    
    # Compare the extrema with the segment's endpoints to find the AABox    
    q1 = coordinates(q[1])
    q2 = coordinates(q[2])
    q3 = coordinates(q[3])
    𝘃₁₃ = q3 - q1
    𝘃₂₃ = q3 - q2
    𝘂 = 3𝘃₁₃ + 𝘃₂₃
    𝘃 = -2(𝘃₁₃ + 𝘃₂₃)
    𝗿 = 𝘂 ⊘ -2𝘃  
    P_stationary = @. q1 + 𝗿 * 𝘂 + 𝗿 * 𝗿 * 𝘃
    P_min = min.(q1, q2)       
    P_max = max.(q1, q2)       
    xmin, ymin = P_min                     
    xmax, ymax = P_max                     
    if 0 < 𝗿[1] < 1                        
        xmin = min(P_min[1], P_stationary[1])    
        xmax = max(P_max[1], P_stationary[1])    
    end    
    if 0 < 𝗿[2] < 1    
        ymin = min(P_min[2], P_stationary[2])    
        ymax = max(P_max[2], P_stationary[2])    
    end    
    return AABox{2, T}(Point{2, T}(xmin, ymin), Point{2, T}(xmax, ymax))    
end

function boundingbox(poly::Polygon{N, Point{D, T}}) where {N, D, T}
    return boundingbox(vertices(poly))::AABox{D, T}
end

function boundingbox(poly::QuadraticPolygon{N, Point{D, T}})::AABox{D, T} where {N, D, T}
    return mapreduce(boundingbox, ∪, edges(poly))
end
