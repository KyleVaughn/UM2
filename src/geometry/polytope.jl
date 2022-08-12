export Polytope, Edge, Face, Cell

export measure

# POLYTOPE    
# -----------------------------------------------------------------------------    
#    
# A K-dimensional polytope, of polynomial order P, represented by the connectivity    
# of its vertices. These N vertices are D-dimensional vertices of type T.    
#    
# This struct only supports the shapes found in "The Visualization Toolkit:    
# An Object-Oriented Approach to 3D Graphics, 4th Edition, Chapter 8, Advanced    
# Data Representation".    
#    
# See the VTK book for specific vertex ordering info, but generally vertices are    
# ordered in a counterclockwise fashion, with vertices of the linear shape given    
# first.    
#    
# See https://en.wikipedia.org/wiki/Polytope for help with terminology.    

abstract type Polytope{K, D, T} end

# 1-polytope
abstract type Edge{D, T} <: Polytope{1, D, T} end

# 2-polytope
abstract type Face{D, T} <: Polytope{2, D, T} end
abstract type Polygon{D, T} <: Face{D, T} end
abstract type QuadraticPolygon{D, T} <: Face{D, T} end

# 3-polytope
abstract type Cell{D, T} <: Polytope{3, D, T} end
abstract type Polyhedron{D, T} <: Cell{D, T} end
abstract type QuadraticPolyhedron{D, T} <: Cell{D, T} end

# Measure
measure(p::Edge) = arclength(p)
measure(p::Face) = area(p)
measure(p::Cell) = volume(p)
