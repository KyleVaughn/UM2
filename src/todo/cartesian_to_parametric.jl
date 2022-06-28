function real_to_parametric(p::Point2D, poly::QuadraticPolygon{N, 2, T}) where {N, T}
    return real_to_parametric(p, poly, 30)
end

function real_to_parametric(p::Point2D{T}, poly::QuadraticPolygon{N, 2, T},
                            max_iters::Int64) where {N, T}
    # Convert from real coordinates to the polygon's local parametric coordinates using
    # Newton-Raphson.
    # If a conversion doesn't exist, the minimizer is returned.
    # Initial guess at polygon centroid
    if N === 6 # Triangle
        rs = SVector{2, T}(1 // 3, 1 // 3)
    else # Quadrilateral
        rs = SVector{2, T}(1 // 2, 1 // 2)
    end
    for i in 1:max_iters
        Δrs = inv(𝗝(poly, rs[1], rs[2])) * (p - poly(rs[1], rs[2]))
        if Δrs ⋅ Δrs < T((1e-8)^2)
            break
        end
        rs += Δrs
    end
    return Point2D{T}(rs[1], rs[2])
end
