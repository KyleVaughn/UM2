#include <um2/gmsh/mesh.hpp>

#if UM2_ENABLE_GMSH

namespace um2::gmsh::model::mesh
{

void
setGlobalMeshSize(double const size)
{
  gmsh::vectorpair dimtags;
  gmsh::model::getEntities(dimtags, 0);
  gmsh::model::mesh::setSize(dimtags, size);
}

void
generateMesh(MeshType const mesh_type, int const opt_iters, int const smooth_iters)
{

  gmsh::option::setNumber("Mesh.SecondOrderIncomplete", 1);
  gmsh::option::setNumber("Mesh.Smoothing", smooth_iters);
  switch (mesh_type) {
  case MeshType::Tri:
    Log::info("Generating triangle mesh");
    // Delaunay (5) handles large element size gradients better. Maybe use that?
    gmsh::option::setNumber("Mesh.Algorithm", 6);
    gmsh::model::mesh::generate(2);
    for (int i = 0; i < opt_iters; ++i) {
      gmsh::model::mesh::optimize("Relocate2D");
      gmsh::model::mesh::optimize("Laplace2D");
    }
    break;
  case MeshType::Quad:
    Log::info("Generating quadrilateral mesh");
    gmsh::option::setNumber("Mesh.RecombineAll", 1);
    gmsh::option::setNumber("Mesh.Algorithm", 8); // Frontal-Delaunay for quads.
    gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", 1);   // All quads
    gmsh::option::setNumber("Mesh.RecombinationAlgorithm", 2); // simple full-quad
    gmsh::model::mesh::generate(2);
    for (int i = 0; i < opt_iters; ++i) {
      //                gmsh::model::mesh::optimize("QuadQuasiStructured");
      gmsh::model::mesh::optimize("Relocate2D");
      gmsh::model::mesh::optimize("Laplace2D");
    }
    break;
  case MeshType::QuadraticTri:
    Log::info("Generating quadratic triangle mesh");
    gmsh::option::setNumber("Mesh.Algorithm", 6);
    gmsh::model::mesh::generate(2);
    gmsh::option::setNumber("Mesh.HighOrderOptimize", 2); // elastic + opt
    gmsh::model::mesh::setOrder(2);
    for (int i = 0; i < opt_iters; ++i) {
      gmsh::model::mesh::optimize("HighOrderElastic");
      gmsh::model::mesh::optimize("Relocate2D");
      gmsh::model::mesh::optimize("HighOrderElastic");
    }
    break;
  case MeshType::QuadraticQuad:
    Log::info("Generating quadratic quadrilateral mesh");
    gmsh::option::setNumber("Mesh.RecombineAll", 1);
    gmsh::option::setNumber("Mesh.Algorithm", 8); // Frontal-Delaunay for quads.
    gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", 1);   // All quads
    gmsh::option::setNumber("Mesh.RecombinationAlgorithm", 2); // simple full-quad
    gmsh::model::mesh::generate(2);
    gmsh::option::setNumber("Mesh.HighOrderOptimize", 2); // elastic + opt
    gmsh::model::mesh::setOrder(2);
    for (int i = 0; i < opt_iters; ++i) {
      gmsh::model::mesh::optimize("HighOrderElastic");
      gmsh::model::mesh::optimize("Relocate2D");
      gmsh::model::mesh::optimize("HighOrderElastic");
    }
    break;
  default:
    Log::error("Invalid mesh type");
  }
}

} // namespace um2::gmsh::model::mesh
#endif // UM2_ENABLE_GMSH
