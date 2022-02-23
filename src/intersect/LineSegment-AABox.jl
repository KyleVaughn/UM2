# Uses a special case of the method in 
# Kay, T. L., & Kajiya, J. T. (1986). Ray tracing complex scenes
#
# Assumes the line passes all the way through the AABox if it intersects, which is a 
# valid assumption for this ray tracing application. 
#
# This version is branchless and is likely faster on the GPU
#function intersect(l::LineSegment, aab::AABox)
#    𝘂⁻¹= 1 ./ l.𝘂   
#    𝘁₁ = 𝘂⁻¹*(aab.minima - l.𝘅₁)
#    𝘁₂ = 𝘂⁻¹*(aab.maxima - l.𝘅₁)
#    tmin = maximum(min.(𝘁₁, 𝘁₂))
#    tmax = minimum(max.(𝘁₁, 𝘁₂))
#    return (tmax >= tmin, SVector(l(tmin), l(tmax)))
#end

# This version has branches and is slightly faster on CPU
# Section 5.3.3 in Ericson, C. (2004). Real-time collision detection
function intersect(l::LineSegment{N,T}, aab::AABox{N,T}) where {N,T}
    tmin = typemin(T)
    tmax = typemax(T)
    p_nan = nan(typeof(l.𝘅₁)) 
    for i = 1:N 
        if abs(l.𝘂[i]) < 1e-6
            if l.𝘅₁[i] < aab.minima[i] || aab.maxima[i] < l.𝘅₁[i]
                return (false, SVector(p_nan, p_nan))
            end
        else
            u⁻¹= 1/l.𝘂[i]
            t₁ = (aab.minima[i] - l.𝘅₁[i])*u⁻¹
            t₂ = (aab.maxima[i] - l.𝘅₁[i])*u⁻¹
            if t₁ > t₂
                t₁,t₂ = t₂,t₁
            end
            tmin = max(tmin, t₁) 
            tmax = min(tmax, t₂) 
            if tmin > tmax
                return (false, SVector(p_nan, p_nan))
            end
        end
    end 
    return (true, SVector(l(tmin), l(tmax)))
end
