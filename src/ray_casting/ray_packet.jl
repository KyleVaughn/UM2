export Ray2Packet

struct Ray2Packet{N, T} 
    origin_x::NTuple{N, T}
    origin_y::NTuple{N, T}
    direction_x::NTuple{N, T}
    direction_y::NTuple{N, T}
end
