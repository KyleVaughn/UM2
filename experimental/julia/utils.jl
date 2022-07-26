using BenchmarkTools
using LinearAlgebra
using Printf
using Test

const line_padding = 64
function format_string(s)
    return lpad(s, line_padding, ' ')
end

function format_time(x)
    if x < 1e-6
        return @sprintf("%8.3g ns", x * 1e9)
    elseif x < 1e-3
        return @sprintf("%8.3g μs", x * 1e6)
    elseif x < 1
        return @sprintf("%8.3g ms", x * 1e3)
    else
        return @sprintf("%8.3g  s", x)
    end
end

function format_comparison(x)
    return @sprintf("%8.3g times faster", x) 
end

function display_results(name, trait1, trait2, time1, time2) 
    println(format_string(repeat(" ", 4)*trait1*" "*name*": "*format_time(time1)))    
    println(format_string(repeat(" ", 4)*trait2*" "*name*": "*format_time(time2)))    
    println(format_string(repeat(" ", 6)*trait1*" is "*format_comparison(time2/time1)))    
    println()
end
