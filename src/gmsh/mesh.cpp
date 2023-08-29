#include <um2/gmsh/mesh.hpp>

#if UM2_USE_GMSH

namespace um2::gmsh::model::mesh
{

//=============================================================================
// setGlobalMeshSize
//=============================================================================

void
setGlobalMeshSize(double const size)
{
  gmsh::vectorpair dimtags;
  gmsh::model::getEntities(dimtags, 0);
  gmsh::model::mesh::setSize(dimtags, size);
}

//=============================================================================
// generateMesh 
//=============================================================================

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

//=============================================================================
// setMeshFieldFromGroups 
//=============================================================================

auto
setMeshFieldFromGroups(    
        int const dim,    
        std::vector<std::string> const & groups,    
        std::vector<double> const & sizes) -> std::vector<int>
{
  // Get all group dimtags for use later
  gmsh::vectorpair dimtags;
  gmsh::model::getPhysicalGroups(dimtags, dim);
  std::vector<int> field_ids(groups.size(), -1);
  // For each of the groups we wish to assign a field to
  for (size_t i = 0; i < groups.size(); ++i) {
    // Create a constant field
    int const fid = gmsh::model::mesh::field::add("Constant");
    field_ids[i] = fid;
    gmsh::model::mesh::field::setNumber(fid, "VIn", sizes[i]);
    // Populate each of the fields with the entities in the
    // physical group
    auto const & group_name = groups[i];
    for (auto const & existing_group_dimtag : dimtags) {
      int const existing_group_tag = existing_group_dimtag.second;
      std::string existing_group_name;
      gmsh::model::getPhysicalName(dim, existing_group_tag, existing_group_name);
      if (group_name == existing_group_name) {
        std::vector<int> tags;
        gmsh::model::getEntitiesForPhysicalGroup(dim, existing_group_tag, tags);
        assert(!tags.empty());
        std::vector<double> double_tags(tags.size());
        for (size_t j = 0; j < tags.size(); j++) {
          double_tags[j] = static_cast<double>(tags[j]);
        }
        switch(dim) {
          case 0:
            gmsh::model::mesh::field::setNumbers(fid, "PointsList", double_tags);
            break;
          case 1:
            gmsh::model::mesh::field::setNumbers(fid, "CurvesList", double_tags);
            break;
          case 2:
            gmsh::model::mesh::field::setNumbers(fid, "SurfacesList", double_tags);
            break;
          case 3:
            gmsh::model::mesh::field::setNumbers(fid, "VolumesList", double_tags);
            break;
          default:
            LOG_ERROR("Invalid dimension");
        } // dim switch
      } // group_name == existing_group_name
    } // existing_group_dimtag : dimtags
    LOG_ERROR("The model does not contain a " + std::to_string(dim) + 
        "-dimensional group with name: " + group_name);
  } // for (size_t i = 0; i < groups.size()) {
  // Create a field that takes the min of each and set as background mesh
  int const fid = gmsh::model::mesh::field::add("Min");
  std::vector<double> double_field_ids(field_ids.size());
  for (size_t j = 0; j < field_ids.size(); ++j) {
    double_field_ids[j] = static_cast<double>(field_ids[j]);
  }
  gmsh::model::mesh::field::setNumbers(fid, "FieldsList", double_field_ids);
  gmsh::model::mesh::field::setAsBackgroundMesh(fid);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);
  return field_ids;
}

} // namespace um2::gmsh::model::mesh
#endif // UM2_USE_GMSH
