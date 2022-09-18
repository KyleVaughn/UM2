export get_modular_rays

# Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html   

# Get modular rays in the AABox2 aab at angle γ for ray spacing s
function get_modular_rays(γ::T, s::T, aab::AABox2{T}) where {T}
    w = delta_x(aab)
    h = delta_y(aab)
    γ1 = γ
    if T(pi_over_2) < γ
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
    if γ ≤ T(pi_over_2)
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
