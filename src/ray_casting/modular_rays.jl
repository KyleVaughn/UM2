export get_modular_rays, get_modular_rays!

function get_modular_ray_params(γ::T, s::T, x_origin::T, y_origin::T, 
                                w::T, h::T) where {T}
    # Number of rays in x and y directions
    sin_γ, cos_γ = sincos(γ)
    nx = ceil(Int64, abs(w * sin_γ / s))
    ny = ceil(Int64, abs(h * cos_γ / s))
    # Effective angle to ensure cyclic rays
    γₑ = atan((h * nx) / (w * ny))
    sin_γₑ, cos_γₑ = sincos(γₑ)

    if γ ≤ T(PI_2)
        dx = -w / nx
        x₀ = x_origin + w - dx / 2
        dir = Vec2{T}(cos_γₑ, sin_γₑ)
    else
        dx = w / nx
        x₀ = x_origin - dx / 2
        dir = Vec2{T}(-cos_γₑ, sin_γₑ)
    end

    dy = h / ny 
    y₀ = y_origin - dy / 2

    return nx, ny, x₀, y₀, dx, dy, dir
end


# Get modular rays in the AABox2 aab at angle γ for ray spacing s
function get_modular_rays(γ::T, s::T, aab::AABox2{T}) where {T}
    # Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html   
    w = width(aab)
    h = height(aab)
    γ1 = γ
    if T(PI_2) < γ
        γ1 = T(π) - γ
    end
    # Number of rays in x and y directions
    nx = ceil(Int64, w * sin(γ1) / s)
    ny = ceil(Int64, h * cos(γ1) / s)
    # Allocate vector for rays
    rays = Vector{Ray2{T}}(undef, nx + ny)
    # Effective angle to ensure cyclic rays
    γₑ = atan((h * nx) / (w * ny))
    # Effective spacing and related quantities
    sin_γₑ, cos_γₑ = sincos(γₑ)
    inv_sin_γₑ = 1 / sin_γₑ
    inv_cos_γₑ = 1 / cos_γₑ
    tan_γₑ = sin_γₑ * inv_cos_γₑ
    inv_tan_γₑ = cos_γₑ * inv_sin_γₑ
    s_eff = w * sin_γₑ / nx
    dx = s_eff * inv_sin_γₑ
    dy = s_eff / cos_γₑ
    xmin = x_min(aab)
    ymin = y_min(aab)
    if γ ≤ T(PI_2)
        for ix in 1:nx
            # Generate ray from the bottom edge of the rectangular domain
            # Ray either terminates at the right edge of the rectangle
            # or on the top edge of the rectangle
            x₀ = w - (ix - T(0.5)) * dx
            y₀ = T(0)
            x₁ = min(w, h * inv_tan_γₑ + x₀)
            y₁ = min((w - x₀) * tan_γₑ, h)
            rays[ix] = Ray2{T}(Point2{T}(xmin + x₀, ymin + y₀),
                               normalize(Vec2{T}(x₁ - x₀, y₁ - y₀)))
        end
        for iy in 1:ny
            # Generate rays from the left edge of the rectangular domain
            # Ray either terminates at the right edge of the rectangle
            # or on the top edge of the rectangle
            x₀ = T(0)
            y₀ = (iy - T(0.5)) * dy
            x₁ = min(w, (h - y₀) * inv_tan_γₑ)
            y₁ = min(w * tan_γₑ + y₀, h)
            rays[nx + iy] = Ray2{T}(Point2{T}(xmin + x₀, ymin + y₀),
                                    normalize(Vec2{T}(x₁ - x₀, y₁ - y₀)))
        end
        return rays
    else
        for ix in 1:nx                                                       
            # Generate ray from the bottom edge of the rectangular domain     
            # Ray either terminates at the left edge of the rectangle
            # or on the top edge of the rectangle
            x₀ = (ix - T(0.5)) * dx
            y₀ = T(0)    
            x₁ = max(0, -h * inv_tan_γₑ + x₀)    
            y₁ = min(x₀ * tan_γₑ, h)      
            rays[ix] = Ray2{T}(Point2{T}(xmin + x₀, ymin + y₀),    
                               normalize(Vec2{T}(x₁ - x₀, y₁ - y₀)))    
        end
        for iy in 1:ny
            # Generate ray from the right edge of the rectangular domain     
            # Segment either terminates at the left edge of the rectangle
            # or on the top edge of the rectangle
            x₀ = w
            y₀ = (iy - T(0.5)) * dy
            x₁ = max(0, w - (h - y₀) * inv_tan_γₑ)
            y₁ = min(w * tan_γₑ + y₀, h)
            rays[nx + iy] = Ray2{T}(Point2{T}(xmin + x₀, ymin + y₀),
                                    normalize(Vec2{T}(x₁ - x₀, y₁ - y₀)))
        end
        return rays
    end
end

# Assumes γ ∈ [0, π]
function get_modular_rays!(rays::Vector{Ray2{T}}, γ::T, s::T, 
                           x_origin::T, y_origin::T, w::T, h::T) where {T}
    (nx, ny, x₀, y₀, dx, dy, dir) = get_modular_ray_params(γ, s, x_origin, 
                                                           y_origin, w, h)
    # Number of rays in x and y directions
    nrays = nx + ny
    # Resize vector for rays if necessary
    if length(rays) < nrays
        resize!(rays, nrays)
    end
    for ix in 1:nx
        rays[ix] = Ray2{T}(Point2{T}(x₀ + ix * dx, y_origin), dir)
    end
    x_end = x₀ + (nx + 0.5) * dx
    for iy in 1:ny
        rays[nx + iy] = Ray2{T}(Point2{T}(x_end, y₀ + iy * dy), dir)
    end
    return nrays
end

# Get the origin points and direction of rays for a given angle γ
function get_modular_rays!(points::Vector{Point2{T}}, γ::T, s::T, 
                           x_origin::T, y_origin::T, w::T, h::T) where {T}
    (nx, ny, x₀, y₀, dx, dy, dir) = get_modular_ray_params(γ, s, x_origin, 
                                                           y_origin, w, h)
    nrays = nx + ny
    # Resize vector for rays if necessary
    if length(points) < nrays
        resize!(points, nrays)
    end
    for ix in 1:nx                                                       
        # Generate ray from the bottom edge of the rectangular domain     
        # Ray either terminates at the left edge of the rectangle
        # or on the top edge of the rectangle
        points[ix] = Point2{T}(x₀ + ix * dx, y_origin)
    end
    x_end = x₀ + (nx + 0.5) * dx
    for iy in 1:ny
        # Generate ray from the right edge of the rectangular domain     
        # Segment either terminates at the left edge of the rectangle
        # or on the top edge of the rectangle
        points[nx + iy] = Point2{T}(x_end, y₀ + iy * dy)
    end
    return nrays, dir
end

function get_modular_rays!(x::Vector{T}, y::Vector{T}, γ::T, s::T, 
                           x_origin::T, y_origin::T, w::T, h::T) where {T}
    (nx, ny, x₀, y₀, dx, dy, dir) = get_modular_ray_params(γ, s, x_origin, 
                                                           y_origin, w, h)
    nrays = nx + ny
    # Resize vector for rays if necessary
    if length(x) < nrays
        resize!(x, nrays)
        resize!(y, nrays)
    end
    for ix in 1:nx
        x[ix] = x₀ + ix * dx
    end
    x_end = x₀ + (nx + 0.5) * dx
    x[(nx + 1):nrays] .= x_end

    y[1:nx] .= y_origin
    for iy in 1:ny
        y[nx + iy] = y₀ + iy * dy
    end

    return nrays, dir
end

function get_modular_rays(ang_quad::ProductAngularQuadrature{T}, 
                          s::T, 
                          aab::AABox2{T}) where {T}
    nγ = length(ang_quad.γ)
    rays = Vector{Vector{Ray2{T}}}(undef, 2 * nγ)
    for i in 1:nγ
        rays[2 * i - 1] = get_modular_rays(       ang_quad.γ[i], s, aab)
        rays[2 * i    ] = get_modular_rays(T(π) - ang_quad.γ[i], s, aab)
    end
    return rays
end
