export Polytope, 
       Edge, 
       LineSegment, 
       QuadraticSegment, 
       Face, 
       Polygon, 
       QuadraticPolygon,
       Triangle, 
       Quadrilateral, 
       QuadraticTriangle, 
       QuadraticQuadrilateral, 
       Cell,
       Polyhedron, 
       QuadraticPolyhedron, 
       Tetrahedron, 
       Hexahedron, 
       QuadraticTetrahedron,
       QuadraticHexahedron

export vertices, 
       facets, 
       ridges, 
       peaks, 
       alias_string, vertextype, paramdim,
       isstraight

struct Polytope{K, P, N, D, T <: AbstractFloat}
    vertices::Vec{N, Point{D, T}}
end

# -- Type aliases --

# 1-polytopes
const Edge             = Polytope{1}

const LineSegment      = Edge{1, 2}
const LineSegment2     = LineSegment{2}
const LineSegment2f    = LineSegment2{Float32}
const LineSegment2d    = LineSegment2{Float64}
const LineSegment2b    = LineSegment2{BigFloat}

const QuadraticSegment = Edge{2, 3}
const QuadraticSegment2 = QuadraticSegment{2}
const QuadraticSegment2f = QuadraticSegment2{Float32}
const QuadraticSegment2d = QuadraticSegment2{Float64}
const QuadraticSegment2b = QuadraticSegment2{BigFloat}

# 2-polytopes
const Face                   = Polytope{2}
const Polygon                = Face{1}
const QuadraticPolygon       = Face{2}

const Triangle               = Polygon{3}
const Triangle2              = Triangle{2}
const Triangle2f             = Triangle2{Float32}
const Triangle2d             = Triangle2{Float64}
const Triangle2b             = Triangle2{BigFloat}

const Quadrilateral          = Polygon{4}
const Quadrilateral2         = Quadrilateral{2}
const Quadrilateral2f        = Quadrilateral2{Float32}
const Quadrilateral2d        = Quadrilateral2{Float64}
const Quadrilateral2b        = Quadrilateral2{BigFloat}

const QuadraticTriangle      = QuadraticPolygon{6}
const QuadraticTriangle2     = QuadraticTriangle{2}
const QuadraticTriangle2f    = QuadraticTriangle2{Float32}
const QuadraticTriangle2d    = Quadrilateral2{Float64}
const QuadraticTriangle2b    = Quadrilateral2{BigFloat}

const QuadraticQuadrilateral = QuadraticPolygon{8}
const QuadraticQuadrilateral2 = QuadraticQuadrilateral{2}
const QuadraticQuadrilateral2f = QuadraticQuadrilateral2{Float32}
const QuadraticQuadrilateral2d = Quadrilateral2{Float64}
const QuadraticQuadrilateral2b = Quadrilateral2{BigFloat}

# 3-polytopes
# All with D = 3.
# const Cell                 = Polytope{3}
# const Polyhedron           = Cell{1}
# const QuadraticPolyhedron  = Cell{2}
# 
# const Tetrahedron          = Polyhedron{4}
# const Hexahedron           = Polyhedron{8}
# const QuadraticTetrahedron = QuadraticPolyhedron{10}
# const QuadraticHexahedron  = QuadraticPolyhedron{20}

# -- Constructors --

# List out the constructors for N vertices to avoid splatting.
# 2 vertices
function Polytope{K, P, 2, D, T}(p1::Point{D, T}, 
                                 p2::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 2, D, T}(Vec{2, Point{D, T}}(p1, p2))
end

# 3 vertices
function Polytope{K, P, 3, D, T}(p1::Point{D, T}, 
                                 p2::Point{D, T}, 
                                 p3::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 3, D, T}(Vec{3, Point{D, T}}(p1, p2, p3))
end

# 4 vertices
function Polytope{K, P, 4, D, T}(p1::Point{D, T}, 
                                 p2::Point{D, T}, 
                                 p3::Point{D, T}, 
                                 p4::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 4, D, T}(Vec{4, Point{D, T}}(p1, p2, p3, p4))
end

# 6 vertices
function Polytope{K, P, 6, D, T}(p1::Point{D, T}, 
                                 p2::Point{D, T}, 
                                 p3::Point{D, T}, 
                                 p4::Point{D, T}, 
                                 p5::Point{D, T}, 
                                 p6::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 6, D, T}(Vec{6, Point{D, T}}(p1, p2, p3, p4, p5, p6))
end

# 8 vertices
function Polytope{K, P, 8, D, T}(p1::Point{D, T}, 
                                 p2::Point{D, T}, 
                                 p3::Point{D, T}, 
                                 p4::Point{D, T}, 
                                 p5::Point{D, T}, 
                                 p6::Point{D, T}, 
                                 p7::Point{D, T}, 
                                 p8::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 8, D, T}(Vec{8, Point{D, T}}(p1, p2, p3, p4, p5, p6, p7, p8))
end

# Constructors for N vertices with implicit type
function Polytope{K, P, 2}(p1::Point{D, T}, 
                           p2::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 2, D, T}(p1, p2)
end

function Polytope{K, P, 3}(p1::Point{D, T}, 
                           p2::Point{D, T}, 
                           p3::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 3, D, T}(p1, p2, p3)
end

function Polytope{K, P, 4}(p1::Point{D, T}, 
                           p2::Point{D, T}, 
                           p3::Point{D, T}, 
                           p4::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 4, D, T}(p1, p2, p3, p4)
end

function Polytope{K, P, 6}(p1::Point{D, T}, 
                           p2::Point{D, T}, 
                           p3::Point{D, T}, 
                           p4::Point{D, T}, 
                           p5::Point{D, T}, 
                           p6::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 6, D, T}(p1, p2, p3, p4, p5, p6)
end

function Polytope{K, P, 8}(p1::Point{D, T}, 
                           p2::Point{D, T}, 
                           p3::Point{D, T}, 
                           p4::Point{D, T}, 
                           p5::Point{D, T}, 
                           p6::Point{D, T}, 
                           p7::Point{D, T}, 
                           p8::Point{D, T}) where {K, P, D, T}
    return Polytope{K, P, 8, D, T}(p1, p2, p3, p4, p5, p6, p7, p8)
end

Polytope{K, P, N}(vertices::Vec{N, Point{D, T}}) where {K, P, N, D, T} = Polytope{K, P, N, D, T}(vertices)

# -- Conversions --

# Convert Vec
function Base.convert(::Type{Polytope{K, P, N, D, T}}, v::Vec{N}) where {K, P, N, D, T}
    return Polytope{K, P, N, D, T}(v...)
end

# -- Accessors --

Base.getindex(poly::Polytope, i::Int) = Base.getindex(poly.vertices, i)

vertices(p::Polytope) = p.vertices

peaks(p::Polytope{3}) = vertices(p)

ridges(p::Polytope{2}) = vertices(p)
ridges(p::Polytope{3}) = edges(p)

facets(p::Polytope{1}) = vertices(p)
facets(p::Polytope{2}) = edges(p)
facets(p::Polytope{3}) = faces(p)

# -- IO --

function Base.show(io::IO, poly::Polytope{K, P, N, D, T}) where {K, P, N, D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    elseif T === BigFloat
        type_char = 'b'
    end
    if K === 1
        if P === 1
            print(io, "LineSegment", D, type_char, "(")
        elseif P === 2
            print(io, "QuadraticSegment", D, type_char, "(")
        else
            error("Not implemented for P > 2")
        end
    elseif K === 2
        if P === 1
            if N === 3
                print(io, "Triangle", D, type_char, "(")
            elseif N === 4
                print(io, "Quadrilateral", D, type_char, "(")
            else
                error("Not implemented for N > 4")
            end
        elseif P === 2
            if N == 6
                print(io, "QuadraticTriangle", D, type_char, "(")
            elseif N == 8
                print(io, "QuadraticQuadrilateral", D, type_char, "(")
            else
                error("Not implemented for N > 8")
            end
        else
            error("Not implemented for P > 2")
        end
    else
        error("Not implemented for K > 2")
    end
    for i in 1:N
        print(io, poly.vertices[i])
        if i < N
            print(io, ", ")
        end
    end
    print(io, ")")
end

## If we think of the polytopes as sets, p₁ ∩ p₂ = p₁ and p₁ ∩ p₂ = p₂ implies p₁ = p₂
#function Base.:(==)(l₁::LineSegment{T}, l₂::LineSegment{T}) where {T}
#    return (l₁[1] === l₂[1] && l₁[2] === l₂[2]) ||
#           (l₁[1] === l₂[2] && l₁[2] === l₂[1])
#end
#Base.:(==)(t₁::Triangle, t₂::Triangle) = return all(v -> v ∈ t₂.vertices, t₁.vertices)
#Base.:(==)(t₁::Tetrahedron, t₂::Tetrahedron) = return all(v -> v ∈ t₂.vertices, t₁.vertices)
#function Base.:(==)(q₁::QuadraticSegment{T}, q₂::QuadraticSegment{T}) where {T}
#    return q₁[3] === q₂[3] &&
#           (q₁[1] === q₂[1] && q₁[2] === q₂[2]) ||
#           (q₁[1] === q₂[2] && q₁[2] === q₂[1])
#end

#isstraight(::LineSegment) = true
#
#"""
#    isstraight(q::QuadraticSegment)
#
#Return if the quadratic segment is effectively straight.
#(If P₃ is at most EPS_POINT distance from LineSegment(P₁,P₂))
#"""
#function isstraight(q::QuadraticSegment{T}) where {T <: Point}
#    # Project P₃ onto the line from P₁ to P₂, call it P₄
#    𝘃₁₃ = q[3] - q[1]
#    𝘃₁₂ = q[2] - q[1]
#    v₁₂ = norm²(𝘃₁₂)
#    𝘃₁₄ = (𝘃₁₃ ⋅ 𝘃₁₂) * inv(v₁₂) * 𝘃₁₂
#    # Determine the distance from P₃ to P₄ (P₄ - P₃ = P₁ + 𝘃₁₄ - P₃ = 𝘃₁₄ - 𝘃₁₃)
#    d² = norm²(𝘃₁₄ - 𝘃₁₃)
#    return d² < T(EPS_POINT^2)
#end
