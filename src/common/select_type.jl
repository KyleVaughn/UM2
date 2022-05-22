function _select_UInt_type(N)
    if N ≤ typemax(UInt8) 
        U = UInt8
    elseif N ≤ typemax(UInt16) 
        U = UInt16
    elseif N ≤ typemax(UInt32) 
        U = UInt32
    elseif N ≤ typemax(UInt64) 
        U = UInt64
    else 
        error(string(N)*" exceeds typemax(UInt64)")
    end
    return U
end
