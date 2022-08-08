# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 
# 4th Edition, Chapter 8, Advanced Data Representation

# 1-polytope
function (p::LineSegment{D, T})(r::T) where {D, T}
    return interpolate_line_segment(p, r)
end

function (p::QuadraticSegment{D, T})(r::T) where {D, T}
    return interpolate_quadratic_segment(p, r)
end

# 2-polytope
function (p::Triangle{D, T})(r::T, s::T) where {D, T}
    return interpolate_triangle(p, r, s)
end

function (p::Quadrilateral{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadrilateral(p, r, s)
end

function (p::QuadraticTriangle{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadratic_triangle(p, r, s)
end

function (p::QuadraticQuadrilateral{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadratic_quadrilateral(p, r, s)
end
