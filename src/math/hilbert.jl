# Chen, Ningtau; Wang, Nengchao; Shi, Baochang
# "A new algorithm for encoding and decoding the Hilbert order," 
# in Software–-Practice and Experience, 2007, 37, 897-908.
function encode_hilbert(x::T, y::T) where {T <: Integer}
    z = T(0)
    if x === T(0) && y === T(0)
        return z
    end

    # floor(log2(max(x, y))) + 1
    rmin = T(1)
    xy_max = max(x, y)
    while (xy_max >>= 1) > 0
        rmin += 1
    end
    
    w = T(1) << (rmin - 1)
    quadrant = T(0)
    while rmin > 0
        if rmin & 1 == 1  # odd
            if x < w
                if y < w # quadrant 0
                    # x, y = x, y
                    quadrant = 0
                else # quadrant 1
                    x, y = y - w, x
                    quadrant = 1
                end
            else
                if y < w # quadrant 3
                    x, y = 2 * w - x - 1, w - y - 1
                    quadrant = 3
                else # quadrant 2
                    x, y = y - w, x - w
                    quadrant = 2
                end
            end
        else  # even
            if x < w
                if y < w # quadrant 0
                    # x, y = x, y
                    quadrant = 0
                else # quadrant 3
                    x, y = w - x - 1, 2 * w - y - 1
                    quadrant = 3
                end
            else
                if y < w # quadrant 1
                    x, y = y, x - w
                    quadrant = 1
                else # quadrant 2
                    x, y = y - w, x - w
                    quadrant = 2
                end
            end
        end
        z = (z << 2) + quadrant
        rmin -= 1
        w >>= 1
    end
    return z
end
