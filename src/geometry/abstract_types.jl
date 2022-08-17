export AbstractEdge,
       AbstractFace,
       AbstractPolygon,
       AbstractLinearPolygon,
       AbstractQuadraticPolygon

# Parametric dimension = 1
abstract type AbstractEdge{D, T <: AbstractFloat} end

# Parametric dimension = 2
abstract type AbstractFace{D, T <: AbstractFloat} end
abstract type AbstractPolygon{D, T} <: AbstractFace{D, T} end
abstract type AbstractLinearPolygon{D, T} <: AbstractPolygon{D, T} end
abstract type AbstractQuadraticPolygon{D, T} <: AbstractPolygon{D, T} end
