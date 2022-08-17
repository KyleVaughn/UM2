abstract type AbstractPolygonMesh{D, T <: AbstractFloat, I <: Integer} end
abstract type AbstractLinearPolygonMesh{D, T, I} <: AbstractPolygonMesh{D, T, I} end
abstract type AbstractQuadraticPolygonMesh{D, T, I} <: AbstractPolygonMesh{D, T, I} end
#-- TriMesh
#-- QuadMesh
#-- TriQuadMesh
#
#-- QuadraticTriMesh
#-- QuadraticQuadMesh
#-- QuadraticTriQuadMesh
