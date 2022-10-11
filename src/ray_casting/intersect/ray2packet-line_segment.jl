function Base.intersect(R::Ray2Packet{N, T}, L::LineSegment2{T}) where {N, T}
    L1x = ntuple(i -> L[1][1], Val(N))
    L1y = ntuple(i -> L[1][2], Val(N))
    L2x = ntuple(i -> L[2][1], Val(N))
    L2y = ntuple(i -> L[2][2], Val(N))
    # 𝘃 = L[2] - L[1]
    𝘃x = @. L2x - L1x
    𝘃y = @. L2y - L1y
    # 𝘂 = R.origin - L[1]
    𝘂x = @. R.origin_x - L1x
    𝘂y = @. R.origin_y - L1y
    # x = 𝘂 × R.direction
    x = @. 𝘂x * R.direction_y - 𝘂y * R.direction_x
    # z = 𝘃 × R.direction
    z = @. 𝘃x * R.direction_y - 𝘃y * R.direction_x
    # y = 𝘂 × 𝘃
    y = @. 𝘂x * 𝘃y - 𝘂y * 𝘃x
    # s = x / z
    s = @. x / z
    # r = y / z
    r = @. y / z
    # 0 ≤ s && s ≤ 1
    valid_s = @. 0 ≤ s && s ≤ 1
    # 0 ≤ s && s ≤ 1 ? r : T(INF_POINT)
    # Use mask to do conditional assignment
    return @. ifelse(valid_s, r, T(INF_POINT))
end
