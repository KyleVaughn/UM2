export boundingbox

function boundingbox(points::Vector{Point{2, T}}) where {T}
    xmin = ymin = typemax(T)
    xmax = ymax = typemin(T)
    for i in 1:length(points)
        x, y = points[i].coord
        xmin = min(xmin, x)
        xmax = max(xmax, x) 
        ymin = 
    end

end
