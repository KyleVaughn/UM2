# Uses a special case of the method in 
# Kay, T. L., & Kajiya, J. T. (1986). Ray tracing complex scenes
#
# Assumes the line passes all the way through the AABox if it intersects, which is a 
# valid assumption for this ray tracing application. 
#
# This version is branchless and is likely faster on the GPU
#function intersect(l::LineSegment, aab::AABox)
#    ๐โปยน= 1 ./ l.๐   
#    ๐โ = ๐โปยน*(aab.minima - l.๐โ)
#    ๐โ = ๐โปยน*(aab.maxima - l.๐โ)
#    tmin = maximum(min.(๐โ, ๐โ))
#    tmax = minimum(max.(๐โ, ๐โ))
#    return (tmax >= tmin, SVector(l(tmin), l(tmax)))
#end

# This version has branches and is slightly faster on CPU
# Section 5.3.3 in Ericson, C. (2004). Real-time collision detection
function intersect(l::LineSegment{N, T}, aab::AABox{N, T}) where {N, T}
    tmin = typemin(T)
    tmax = typemax(T)
    p_nan = nan(typeof(l.๐โ))
    for i in 1:N
        if abs(l.๐[i]) < 1e-6
            if l.๐โ[i] < aab.minima[i] || aab.maxima[i] < l.๐โ[i]
                return (false, SVector(p_nan, p_nan))
            end
        else
            uโปยน = 1 / l.๐[i]
            tโ = (aab.minima[i] - l.๐โ[i]) * uโปยน
            tโ = (aab.maxima[i] - l.๐โ[i]) * uโปยน
            if tโ > tโ
                tโ, tโ = tโ, tโ
            end
            tmin = max(tmin, tโ)
            tmax = min(tmax, tโ)
            if tmin > tmax
                return (false, SVector(p_nan, p_nan))
            end
        end
    end
    return (true, SVector(l(tmin), l(tmax)))
end
