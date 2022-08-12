function ⪇(x::F, y::F) where {F <: AbstractFloat}
    return x < y && eps(F) < abs(x - y)
end

function ⪉(x::F, y::F) where {F <: AbstractFloat}
    return x < y && x ≉ y
end
