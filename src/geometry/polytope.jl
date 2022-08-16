export Polytope, Edge, Face, Cell

export measure,
       edge,
       edges

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

abstract type AbstractPolytope{K, D, T <: AbstractFloat} end

# 1-polytope
abstract type AbstractEdge{D, T} <: AbstractPolytope{1, D, T} end

# 2-polytope
abstract type AbstractFace{D, T} <: AbstractPolytope{2, D, T} end
abstract type AbstractPolygon{D, T} <: AbstractFace{D, T} end
abstract type AbstractQuadraticPolygon{D, T} <: AbstractFace{D, T} end

# 3-polytope
abstract type AbstractCell{D, T} <: AbstractPolytope{3, D, T} end
abstract type AbstractPolyhedron{D, T} <: AbstractCell{D, T} end
abstract type AbstractQuadraticPolyhedron{D, T} <: AbstractCell{D, T} end
