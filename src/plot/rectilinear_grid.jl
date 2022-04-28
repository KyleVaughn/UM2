function convert_arguments(LS::Type{<:LineSegments}, 
                           rg::RectilinearGrid{X,Y,0,T}) where {X,Y,T}
    lines = Vector{LineSegment{Point{2,T}}}(undef, X+Y)
    ylower = ymin(rg)
    yupper = ymax(rg)
    xlower = xmin(rg)
    xupper = xmax(rg)
    for i = 1:X
        lines[i] = LineSegment(Point(rg.x[i], ylower), Point(rg.x[i], yupper))
    end
    for i = 1:Y
        lines[X+i] = LineSegment(Point(xlower, rg.y[i]), Point(xupper, rg.y[i]))
    end
    return convert_arguments(LS, lines)
end

# 3D grid plot quadrilaterals
function convert_arguments(LS::Type{<:LineSegments}, 
                           rg::RectilinearGrid{X,Y,Z,T}) where {X,Y,Z,T}
    lines = Vector{LineSegment{Point{3,T}}}(undef, Z*(X+Y) + Y*(X+Z))
    
    xlower = xmin(rg)
    xupper = xmax(rg)
    ylower = ymin(rg)
    yupper = ymax(rg)
    zlower = zmin(rg)
    zupper = zmax(rg)
    ictr = 1
    # XY planes
    for iz = 1:Z
        for i = 1:X
            lines[ictr] = LineSegment(Point(rg.x[i], ylower, rg.z[iz]), 
                                      Point(rg.x[i], yupper, rg.z[iz]))
            ictr += 1
        end
        for i = 1:Y
            lines[ictr] = LineSegment(Point(xlower, rg.y[i], rg.z[iz]), 
                                      Point(xupper, rg.y[i], rg.z[iz]))
            ictr += 1
        end
    end
    # XZ planes
    for iy = 1:Y
        for i = 1:X
            lines[ictr] = LineSegment(Point(rg.x[i], rg.y[iy], zlower), 
                                      Point(rg.x[i], rg.y[iy], zupper))
            ictr += 1
        end
        for i = 1:Z
            lines[ictr] = LineSegment(Point(xlower, rg.y[iy], rg.z[i]), 
                                      Point(xupper, rg.y[iy], rg.z[i]))
            ictr += 1
        end
    end

    return convert_arguments(LS, lines)
end
