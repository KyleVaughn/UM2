#include <um2/gmsh/mesh.hpp>

#if UM2_USE_GMSH

#  include <um2/math/stats.hpp>

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

// generateMesh(MeshType const mesh_type, int const opt_iters, int const smooth_iters)
void
generateMesh(MeshType const mesh_type, int const smooth_iters)
{

  gmsh::option::setNumber("Mesh.SecondOrderIncomplete", 1);
  gmsh::option::setNumber("Mesh.Smoothing", smooth_iters);
  switch (mesh_type) {
  case MeshType::Tri:
    Log::info("Generating triangle mesh");
    // Delaunay (5) handles large element size gradients better. Maybe use that?
    gmsh::option::setNumber("Mesh.Algorithm", 6);
    gmsh::model::mesh::generate(2);
    //    for (int i = 0; i < opt_iters; ++i) {
    //      gmsh::model::mesh::optimize("Relocate2D");
    //      gmsh::model::mesh::optimize("Laplace2D");
    //    }
    break;
  case MeshType::Quad:
    Log::info("Generating quadrilateral mesh");
    gmsh::option::setNumber("Mesh.RecombineAll", 1);
    gmsh::option::setNumber("Mesh.Algorithm", 8); // Frontal-Delaunay for quads.
    gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", 1);   // All quads
    gmsh::option::setNumber("Mesh.RecombinationAlgorithm", 2); // simple full-quad
    gmsh::model::mesh::generate(2);
    //    for (int i = 0; i < opt_iters; ++i) {
    //      //                gmsh::model::mesh::optimize("QuadQuasiStructured");
    //      gmsh::model::mesh::optimize("Relocate2D");
    //      gmsh::model::mesh::optimize("Laplace2D");
    //    }
    break;
  case MeshType::QuadraticTri:
    Log::info("Generating quadratic triangle mesh");
    gmsh::option::setNumber("Mesh.Algorithm", 6);
    gmsh::model::mesh::generate(2);
    gmsh::option::setNumber("Mesh.HighOrderOptimize", 2); // elastic + opt
    gmsh::model::mesh::setOrder(2);
    //    for (int i = 0; i < opt_iters; ++i) {
    //      gmsh::model::mesh::optimize("HighOrderElastic");
    //      gmsh::model::mesh::optimize("Relocate2D");
    //      gmsh::model::mesh::optimize("HighOrderElastic");
    //    }
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
    //    for (int i = 0; i < opt_iters; ++i) {
    //      gmsh::model::mesh::optimize("HighOrderElastic");
    //      gmsh::model::mesh::optimize("Relocate2D");
    //      gmsh::model::mesh::optimize("HighOrderElastic");
    //    }
    break;
  default:
    Log::error("Invalid mesh type");
  }
}

//=============================================================================
// setMeshFieldFromGroups
//=============================================================================

auto
setMeshFieldFromGroups(int const dim, std::vector<std::string> const & groups,
                       std::vector<double> const & sizes) -> int
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
    bool found = false;
    auto const & group_name = groups[i];
    for (auto const & existing_group_dimtag : dimtags) {
      int const existing_group_tag = existing_group_dimtag.second;
      std::string existing_group_name;
      gmsh::model::getPhysicalName(dim, existing_group_tag, existing_group_name);
      if (group_name == existing_group_name) {
        std::vector<int> tags;
        gmsh::model::getEntitiesForPhysicalGroup(dim, existing_group_tag, tags);
        ASSERT(!tags.empty());
        std::vector<double> double_tags(tags.size());
        for (size_t j = 0; j < tags.size(); j++) {
          double_tags[j] = static_cast<double>(tags[j]);
        }
        switch (dim) {
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
        found = true;
        break;
      } // group_name == existing_group_name
    }   // existing_group_dimtag : dimtags
    if (!found) {
      LOG_ERROR("The model does not contain a ", dim,
                "-dimensional group with name: ", group_name); 
    }
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
  return fid;
}

//=============================================================================
// setMeshFieldFromKnudsenNumber
//=============================================================================

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
setMeshFieldFromKnudsenNumber(int const dim, std::vector<Material> const & materials,
                              double const kn_target, double const mfp_threshold,
                              double const mfp_scale, std::vector<int> const & is_fuel,
                              XSecReduction const strategy) -> int
{
  //-----------------------------------------------------------------------
  // Check that each material exists as a physical group
  //-----------------------------------------------------------------------
  // Get the names of all "Material_XXX" physical groups
  gmsh::vectorpair dimtags;
  gmsh::model::getPhysicalGroups(dimtags, dim);
  std::vector<std::string> material_group_names;
  std::vector<int> material_group_tags;
  for (auto const & dimtag : dimtags) {
    int const tag = dimtag.second;
    std::string name;
    gmsh::model::getPhysicalName(dim, tag, name);
    if (name.starts_with("Material")) {
      material_group_names.push_back(name);
      material_group_tags.push_back(tag);
    }
  }

  // Check that we have the correct number of materials
  size_t const num_material_groups = material_group_names.size();
  size_t const num_materials = materials.size();
  if (num_material_groups != num_materials) {
    LOG_ERROR("Number of materials does not match number of material physical groups");
  }

  // Get the names of all materials in the form of "Material_XXX"
  std::vector<std::string> material_names(num_materials);
  for (size_t i = 0; i < num_materials; ++i) {
    material_names[i] = std::string("Material_") + materials[i].getName().c_str();
  }

  // Check that the names match
  std::vector<int8_t> found(num_materials, 0);
  for (auto const & material_name : material_names) {
    for (size_t i = 0; i < num_materials; ++i) {
      if (material_name == material_group_names[i]) {
        found[i] = 1;
        break;
      }
    }
  }
  for (size_t i = 0; i < num_materials; ++i) {
    if (found[i] == 0) {
      LOG_ERROR(materials[i].getName(), " does not exist as a physical group");
    }
  }

  if (kn_target <= 0.0) {
    LOG_ERROR("Knudsen number must be positive");
  }

  // Compute the target characteristic length for each material
  std::vector<double> lcs(num_materials, 0.0);
  std::vector<double> sigmas_t(num_materials, 0.0);
  // If using average MFP, we need the average Sigma_t for each material
  if (strategy == XSecReduction::Mean) {
    LOG_INFO("Computing Knudsen number using groupwise average mean free path");
  } else if (strategy == XSecReduction::Max) {
    LOG_INFO("Computing Knudsen number using groupwise minimum mean free path");
  } else {
    LOG_ERROR("Invalid Knudsen number computation strategy");
  }

  for (size_t i = 0; i < num_materials; ++i) {
    double const sigma_t = materials[i].xs().getOneGroupTotalXS(strategy);
    // The mean chord length of an equilateral triangle with side length l:
    // s = pi * A/ 3l = pi * sqrt(3) * l / 12
    //
    // Kn = MFP / MCL = 1 / (sigma_t * s)
    //
    // s = 1 / (sigma_t * Kn)
    // l = 12 / (sigma_t * kn * sqrt(3) * pi)
    lcs[i] = 12.0 / (um2::pi<double> * std::sqrt(3.0) * kn_target * sigma_t);
    sigmas_t[i] = sigma_t;
  }

  // Create the base fields, which are constant in each material
  std::vector<int> field_ids(num_materials, -1);
  for (size_t i = 0; i < num_materials; ++i) {
    // Create a constant field
    int const fid = gmsh::model::mesh::field::add("Constant");
    field_ids[i] = fid;
    gmsh::model::mesh::field::setNumber(fid, "VIn", lcs[i]);
    // Populate each of the fields with the entities in the
    // physical group
    std::vector<int> ent_tags;
    gmsh::model::getEntitiesForPhysicalGroup(dim, material_group_tags[i], ent_tags);
    ASSERT(!ent_tags.empty());
    std::vector<double> const double_ent_tags(ent_tags.begin(), ent_tags.end());
    switch (dim) {
    case 0:
      gmsh::model::mesh::field::setNumbers(fid, "PointsList", double_ent_tags);
      break;
    case 1:
      gmsh::model::mesh::field::setNumbers(fid, "CurvesList", double_ent_tags);
      break;
    case 2:
      gmsh::model::mesh::field::setNumbers(fid, "SurfacesList", double_ent_tags);
      break;
    case 3:
      gmsh::model::mesh::field::setNumbers(fid, "VolumesList", double_ent_tags);
      break;
    default:
      LOG_ERROR("Invalid dimension");
    } // dim switch
  }   // for (size_t i = 0; i < num_materials; ++i)

  // We now have the target characteristic length for each material. If
  // mfp_threshold > 0.0, then for non-fuel materials, we wish to scale
  // the characteristic length by the distance to the nearest fuel material,
  // after some threshold MFPs
  if (mfp_threshold > 0.0) {
    ASSERT(mfp_scale > 0.0);
    // First, get all the entities in the fuel materials
    std::vector<int> fuel_ent_tags;
    for (size_t i = 0; i < num_materials; ++i) {
      if (is_fuel[i] == 1) {
        std::vector<int> ent_tags;
        gmsh::model::getEntitiesForPhysicalGroup(dim, material_group_tags[i], ent_tags);
        fuel_ent_tags.insert(fuel_ent_tags.end(), ent_tags.begin(), ent_tags.end());
      }
    }
    // Ensure there are no duplicates
    std::sort(fuel_ent_tags.begin(), fuel_ent_tags.end());
    for (size_t i = 0; i < fuel_ent_tags.size() - 1; ++i) {
      if (fuel_ent_tags[i] == fuel_ent_tags[i + 1]) {
        LOG_ERROR("Duplicate entity tag in fuel material physical group");
      }
    }
    std::vector<double> const fuel_ent_tags_d(fuel_ent_tags.begin(), fuel_ent_tags.end());

    // Create the fuel distance field
    int const fuel_distance_fid = gmsh::model::mesh::field::add("Distance");
    switch (dim) {
    case 0:
      gmsh::model::mesh::field::setNumbers(fuel_distance_fid, "PointsList",
                                           fuel_ent_tags_d);
      break;
    case 1:
      gmsh::model::mesh::field::setNumbers(fuel_distance_fid, "CurvesList",
                                           fuel_ent_tags_d);
      break;
    case 2:
      gmsh::model::mesh::field::setNumbers(fuel_distance_fid, "SurfacesList",
                                           fuel_ent_tags_d);
      break;
    case 3:
      gmsh::model::mesh::field::setNumbers(fuel_distance_fid, "VolumesList",
                                           fuel_ent_tags_d);
      break;
    default:
      LOG_ERROR("Invalid dimension");
    } // dim switch

    std::string const fuel_distance_name = 'F' + std::to_string(fuel_distance_fid);
    // For each non-fuel material create a field:
    //          /
    // l       /
    // ^      /
    // x-----x-----------------> lc
    // |    /|
    // |   / |
    // |  /  |
    // | /   |
    // |-----x-----------------> sigma_t * d_fuel (mfps)
    // 0   mfp_threshold
    //
    // lc_linear = lc * sigma_t * mfp_scale * d_fuel +  lc * (1 - mfp_threshold *
    // mfp_scale) Then, create a field which takes the max(lc, lc_linear)
    for (size_t i = 0; i < num_materials; ++i) {
      if (is_fuel[i] == 0) {
        // Create a field that multiplies the base field by the distance to the fuel
        // material
        int const fid = gmsh::model::mesh::field::add("MathEval");
        double const scale = lcs[i] * sigmas_t[i] * mfp_scale;
        double const offset = lcs[i] * (1.0 - mfp_threshold * mfp_scale);
        ASSERT(offset < 0.0);
        std::string const math_expr = fuel_distance_name + " * " + std::to_string(scale) +
                                      " " + std::to_string(offset);
        LOG_INFO("Creating linear field for " + materials[i].name() + ": " +
                 String(math_expr.c_str()));
        gmsh::model::mesh::field::setString(fid, "F", math_expr);
        int const max_fid = gmsh::model::mesh::field::add("Max");
        std::vector<double> const field_ids_d = {static_cast<double>(field_ids[i]),
                                                 static_cast<double>(fid)};
        gmsh::model::mesh::field::setNumbers(max_fid, "FieldsList", field_ids_d);
        // Replace the base field with the max field
        field_ids[i] = max_fid;
      }
    }
  } // if (mfp_threshold > 0.0)

  // Create a field that takes the min of each and set as background mesh
  int const fid = gmsh::model::mesh::field::add("Min");
  std::vector<double> const double_field_ids(field_ids.begin(), field_ids.end());
  gmsh::model::mesh::field::setNumbers(fid, "FieldsList", double_field_ids);
  gmsh::model::mesh::field::setAsBackgroundMesh(fid);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

  return fid;
}

} // namespace um2::gmsh::model::mesh
#endif // UM2_USE_GMSH
