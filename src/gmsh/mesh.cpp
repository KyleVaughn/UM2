#include <um2/config.hpp>

#if UM2_USE_GMSH

#  include <um2/common/logger.hpp>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/gmsh/mesh.hpp>
#  include <um2/mesh/element_types.hpp>
#  include <um2/physics/cross_section.hpp>
#  include <um2/physics/material.hpp>
#  include <um2/stdlib/assert.hpp>
#  include <um2/stdlib/math/roots.hpp>
#  include <um2/stdlib/numbers.hpp>
#  include <um2/stdlib/vector.hpp>

#  include <algorithm>
#  include <cstddef>
#  include <string>
#  include <vector>

namespace um2::gmsh::model::mesh
{

//=============================================================================
// setGlobalMeshSize
//=============================================================================

void
setGlobalMeshSize(double const size)
{
  LOG_INFO("Setting global mesh size to: ", size);
  gmsh::vectorpair dimtags;
  gmsh::model::getEntities(dimtags, 0);
  gmsh::model::mesh::setSize(dimtags, size);
}

//=============================================================================
// generateMesh
//=============================================================================

void
generateMesh(MeshType const mesh_type, int const smooth_iters)
{
  gmsh::option::setNumber("Mesh.SecondOrderIncomplete", 1);
  gmsh::option::setNumber("Mesh.Smoothing", smooth_iters);
  //  Int constexpr mesh_from_curvature = 12;
  switch (mesh_type) {
  case MeshType::Tri:
    LOG_INFO("Generating triangle mesh");
    // Delaunay (5) handles large element size gradients better. Maybe use that?
    gmsh::option::setNumber("Mesh.Algorithm", 5);
    gmsh::model::mesh::generate(2);
    break;
  case MeshType::Quad:
    LOG_INFO("Generating quadrilateral mesh");
    gmsh::option::setNumber("Mesh.RecombineAll", 1);
    gmsh::option::setNumber("Mesh.Algorithm", 8); // Frontal-Delaunay for quads.
    gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", 1);   // All quads
    gmsh::option::setNumber("Mesh.RecombinationAlgorithm", 2); // simple full-quad
    gmsh::model::mesh::generate(2);
    break;
  case MeshType::QuadraticTri:
    LOG_INFO("Generating quadratic triangle mesh");
    gmsh::option::setNumber("Mesh.Algorithm", 5);
    //    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", mesh_from_curvature);
    //    LOG_WARN("Setting mesh size from curvature to ", mesh_from_curvature);
    gmsh::model::mesh::generate(2);
    gmsh::option::setNumber("Mesh.HighOrderOptimize", 2); // elastic + opt
    gmsh::model::mesh::setOrder(2);
    break;
  case MeshType::QuadraticQuad:
    LOG_INFO("Generating quadratic quadrilateral mesh");
    gmsh::option::setNumber("Mesh.RecombineAll", 1);
    gmsh::option::setNumber("Mesh.Algorithm", 8); // Frontal-Delaunay for quads.
    //    gmsh::option::setNumber("Mesh.SubdivisionAlgorithm", 1);   // All quads
    gmsh::option::setNumber("Mesh.RecombinationAlgorithm", 2); // simple full-quad
    gmsh::model::mesh::generate(2);
    gmsh::option::setNumber("Mesh.HighOrderOptimize", 2); // elastic + opt
    gmsh::model::mesh::setOrder(2);
    break;
  default:
    LOG_ERROR("Invalid mesh type");
  }
}

////=============================================================================
//// setMeshFieldFromGroups
////=============================================================================
//
// auto
// setMeshFieldFromGroups(int const dim, std::vector<std::string> const & groups,
//                       std::vector<double> const & sizes) -> int
//{
//  // Get all group dimtags for use later
//  gmsh::vectorpair dimtags;
//  gmsh::model::getPhysicalGroups(dimtags, dim);
//  std::vector<int> field_ids(groups.size(), -1);
//  // For each of the groups we wish to assign a field to
//  for (size_t i = 0; i < groups.size(); ++i) {
//    // Create a constant field
//    int const fid = gmsh::model::mesh::field::add("Constant");
//    field_ids[i] = fid;
//    gmsh::model::mesh::field::setNumber(fid, "VIn", sizes[i]);
//    // Populate each of the fields with the entities in the
//    // physical group
//    bool found = false;
//    auto const & group_name = groups[i];
//    for (auto const & existing_group_dimtag : dimtags) {
//      int const existing_group_tag = existing_group_dimtag.second;
//      std::string existing_group_name;
//      gmsh::model::getPhysicalName(dim, existing_group_tag, existing_group_name);
//      if (group_name == existing_group_name) {
//        std::vector<int> tags;
//        gmsh::model::getEntitiesForPhysicalGroup(dim, existing_group_tag, tags);
//        ASSERT(!tags.empty());
//        std::vector<double> double_tags(tags.size());
//        for (size_t j = 0; j < tags.size(); j++) {
//          double_tags[j] = static_cast<double>(tags[j]);
//        }
//        switch (dim) {
//        case 0:
//          gmsh::model::mesh::field::setNumbers(fid, "PointsList", double_tags);
//          break;
//        case 1:
//          gmsh::model::mesh::field::setNumbers(fid, "CurvesList", double_tags);
//          break;
//        case 2:
//          gmsh::model::mesh::field::setNumbers(fid, "SurfacesList", double_tags);
//          break;
//        case 3:
//          gmsh::model::mesh::field::setNumbers(fid, "VolumesList", double_tags);
//          break;
//        default:
//          LOG_ERROR("Invalid dimension");
//        } // dim switch
//        found = true;
//        break;
//      } // group_name == existing_group_name
//    }   // existing_group_dimtag : dimtags
//    if (!found) {
//      LOG_ERROR("The model does not contain a ", dim,
//                "-dimensional group with name: ", group_name);
//    }
//  } // for (size_t i = 0; i < groups.size()) {
//  // Create a field that takes the min of each and set as background mesh
//  int const fid = gmsh::model::mesh::field::add("Min");
//  std::vector<double> double_field_ids(field_ids.size());
//  for (size_t j = 0; j < field_ids.size(); ++j) {
//    double_field_ids[j] = static_cast<double>(field_ids[j]);
//  }
//  gmsh::model::mesh::field::setNumbers(fid, "FieldsList", double_field_ids);
//  gmsh::model::mesh::field::setAsBackgroundMesh(fid);
//  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
//  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
//  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);
//  return fid;
//}
//
//=============================================================================
// setMeshFieldFromKnudsenNumber
//=============================================================================

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
setMeshFieldFromKnudsenNumber(int const dim, um2::Vector<Material> const & materials,
                              double const kn_target, double const fuel_mfp_threshold,
                              double const fuel_mfp_scale, double const abs_mfp_threshold,
                              double const abs_mfp_scale) -> int
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
  auto const num_materials = static_cast<size_t>(materials.size());
  if (num_material_groups != num_materials) {
    LOG_ERROR("Number of materials does not match number of material physical groups");
    return -1;
  }

  // Get the names of all materials in the form of "Material_XXX"
  std::vector<std::string> material_names(num_materials);
  for (size_t i = 0; i < num_materials; ++i) {
    material_names[i] =
        std::string("Material_") + materials[static_cast<Int>(i)].getName().data();
  }

  // Check that the names match
  std::vector<size_t> index_of_material_in_gmsh(num_materials, 10000);
  for (size_t imat = 0; imat < num_materials; ++imat) {
    auto const & material_name = material_names[imat];
    for (size_t igmsh = 0; igmsh < num_materials; ++igmsh) {
      if (material_name == material_group_names[igmsh]) {
        index_of_material_in_gmsh[imat] = igmsh;
        break;
      }
    }
  }
  for (size_t i = 0; i < num_materials; ++i) {
    if (index_of_material_in_gmsh[i] == 10000) {
      LOG_ERROR(materials[static_cast<Int>(i)].getName(),
                " does not exist as a physical group");
      return -1;
    }
  }

  if (kn_target <= 0.0) {
    LOG_ERROR("Knudsen number must be positive");
    return -1;
  }

  // Ensure each material has a valic cross section
  for (auto const & material : materials) {
    if (!material.hasXSec()) {
      LOG_ERROR("Material ", material.getName(), " does not have a valid cross section");
      return -1;
    }
    material.validateXSec(); // Only run if UM2_ENABLE_ASSERTS
  }

  // Compute the target characteristic length for each material
  std::vector<double> lcs(num_materials, 0.0);
  std::vector<double> sigmas_t(num_materials, 0.0);
  std::vector<int> is_fissile(num_materials, 0);
  std::vector<int> is_absorber(num_materials, 0);
  for (size_t i = 0; i < num_materials; ++i) {
    XSec const xs_avg_1g = materials[static_cast<Int>(i)].xsec().collapseTo1GroupAvg();
    ASSERT(xs_avg_1g.t(0) > 0);
    double const sigma_t = xs_avg_1g.t(0);
    double const sigma_s = xs_avg_1g.s()[0];
    double const c = sigma_s / sigma_t;
    // The mean chord length of an equilateral triangle with side length l:
    // s = pi * A / 3l = pi * (sqrt(3) * l^2 / 4) / (3l) = pi * sqrt(3) * l / 12
    //
    // Kn = MFP / MCL = 1 / (sigma_t * s)
    //
    // s = 1 / (sigma_t * Kn)
    //
    // pi * sqrt(3) * l / 12 = 1 / (sigma_t * Kn)
    // l = 12 / (sigma_t * Kn * sqrt(3) * pi)
    // NOLINTNEXTLINE(modernize-use-std-numbers)
    lcs[i] = 12.0 / (um2::pi<double> * um2::sqrt(3.0) * kn_target * sigma_t);
    sigmas_t[i] = sigma_t;
    if (materials[static_cast<Int>(i)].xsec().isFissile()) {
      LOG_INFO("Material ", materials[static_cast<Int>(i)].getName(), " is fissile");
      is_fissile[i] = 1;
    }
    // If c < 0.2, and sigma_t > 0.9, classify as absorber
    if (c < 0.2 && sigma_t > 0.9) {
      LOG_INFO("Material ", materials[static_cast<Int>(i)].getName(), " is an absorber");
      is_absorber[i] = 1;
    }
    auto const & material_name = materials[static_cast<Int>(i)].getName();
    LOG_INFO("Material ", material_name, ": lc = ", lcs[i], " cm, sigma_t = ", sigma_t,
             " cm^-1");
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
    gmsh::model::getEntitiesForPhysicalGroup(
        dim, material_group_tags[index_of_material_in_gmsh[i]], ent_tags);
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
  } // for (size_t i = 0; i < num_materials; ++i)

  // We now have the target characteristic length for each material. If
  // fuel_mfp_threshold >= 0.0, then for non-fuel, non-absorber materials, we wish to
  // scale the characteristic length by the distance to the nearest fuel material, after
  // some threshold MFPs
  if (fuel_mfp_threshold >= 0.0) {
    LOG_INFO("Using fuel MFP threshold: ", fuel_mfp_threshold,
             " and MFP scale: ", fuel_mfp_scale);
    ASSERT(fuel_mfp_scale >= 1.0);
    ASSERT(is_fissile.size() == num_materials);
    // ASSERT that there is at least one fuel material
    if (std::all_of(is_fissile.begin(), is_fissile.end(),
                    [](int const x) { return x == 0; })) {
      LOG_ERROR("No fuel materials found");
      return -1;
    }
    // First, get all the entities in the fuel materials
    std::vector<int> fuel_ent_tags;
    for (size_t i = 0; i < num_materials; ++i) {
      if (is_fissile[i] == 1) {
        std::vector<int> ent_tags;
        gmsh::model::getEntitiesForPhysicalGroup(
            dim, material_group_tags[index_of_material_in_gmsh[i]], ent_tags);
        fuel_ent_tags.insert(fuel_ent_tags.end(), ent_tags.begin(), ent_tags.end());
      }
    }
    // Ensure there are no duplicates
    std::sort(fuel_ent_tags.begin(), fuel_ent_tags.end());
    for (size_t i = 0; i < fuel_ent_tags.size() - 1; ++i) {
      if (fuel_ent_tags[i] == fuel_ent_tags[i + 1]) {
        LOG_ERROR("Duplicate entity tag in fuel material physical group");
        return -1;
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
      return -1;
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
    // 0   fuel_mfp_threshold
    //
    // lc_linear = lc * sigma_t * fuel_mfp_scale * d_fuel +
    //    lc * (1 - fuel_mfp_threshold * fuel_mfp_scale)
    // Then, create a field which takes the max(lc, lc_linear)
    for (size_t i = 0; i < num_materials; ++i) {
      if (is_fissile[i] == 0 && is_absorber[i] == 0) {
        // Create a field that multiplies the base field by the distance to the fuel
        // material
        int const fid = gmsh::model::mesh::field::add("MathEval");
        double const scale = lcs[i] * sigmas_t[i] * fuel_mfp_scale;
        double const offset = lcs[i] * (1.0 - fuel_mfp_threshold * fuel_mfp_scale);
        std::string math_expr = fuel_distance_name + " * " + std::to_string(scale);
        if (offset >= 0.0) {
          math_expr += " + " + std::to_string(offset);
        } else {
          math_expr += " - " + std::to_string(-offset);
        }
        LOG_INFO("Creating linear field for ", materials[static_cast<Int>(i)].getName(),
                 ": ", math_expr.c_str());
        gmsh::model::mesh::field::setString(fid, "F", math_expr);
        int const max_fid = gmsh::model::mesh::field::add("Max");
        std::vector<double> const field_ids_d = {static_cast<double>(field_ids[i]),
                                                 static_cast<double>(fid)};
        gmsh::model::mesh::field::setNumbers(max_fid, "FieldsList", field_ids_d);
        // Replace the base field with the max field
        field_ids[i] = max_fid;
      }
    }
  } // if (fuel_mfp_threshold > 0.0)

  // If abs_mfp_threshold >= 0.0, then for absorber materials, we wish to scale
  // the characteristic length by the distance to the boundary of the absorber
  // after some threshold MFPs
  if (abs_mfp_threshold >= 0.0) {
    LOG_INFO("Using abs MFP threshold: ", abs_mfp_threshold,
             " and MFP scale: ", abs_mfp_scale);
    ASSERT(abs_mfp_scale >= 1.0);
    ASSERT(is_absorber.size() == num_materials);
    // ASSERT that there is at least one absorber material
    if (std::all_of(is_absorber.begin(), is_absorber.end(),
                    [](int const x) { return x == 0; })) {
      LOG_ERROR("No absorber materials found");
      return -1;
    }

    std::vector<int> abs_ent_tags;
    std::vector<int> abs_ent_tags_d1;
    std::vector<int> upward;
    std::vector<int> downward;
    for (size_t i = 0; i < num_materials; ++i) {
      if (is_absorber[i] == 1) {
        abs_ent_tags.resize(0);
        gmsh::model::getEntitiesForPhysicalGroup(
            dim, material_group_tags[index_of_material_in_gmsh[i]], abs_ent_tags);

        abs_ent_tags_d1.resize(0);
        // For each entity in abs_ent_tags
        for (auto const & abs_ent_tag : abs_ent_tags) {
          // Get the D-1 entities
          upward.resize(0);
          downward.resize(0);
          gmsh::model::getAdjacencies(dim, abs_ent_tag, upward, downward);
          ASSERT(!downward.empty());
          std::sort(downward.begin(), downward.end());
          // Ensure there are no duplicates
          for (size_t ient = 0; ient < downward.size() - 1; ++ient) {
            if (downward[ient] == downward[ient + 1]) {
              LOG_ERROR("Duplicate downward entity tag");
              return -1;
            }
          }
          // Insert the downward entities into abs_ent_tags_d1
          for (auto const & d1 : downward) {
            if (std::find(abs_ent_tags_d1.begin(), abs_ent_tags_d1.end(), d1) ==
                abs_ent_tags_d1.end()) {
              abs_ent_tags_d1.push_back(d1);
            }
          }
        } // for (auto const & abs_ent_tag : abs_ent_tags)
        std::sort(abs_ent_tags_d1.begin(), abs_ent_tags_d1.end());
        std::vector<double> const boundary_tags(abs_ent_tags_d1.begin(),
                                                abs_ent_tags_d1.end());

        // Create the abs distance field
        int const abs_distance_fid = gmsh::model::mesh::field::add("Distance");
        switch (dim - 1) {
        case 0:
          gmsh::model::mesh::field::setNumbers(abs_distance_fid, "PointsList",
                                               boundary_tags);
          break;
        case 1:
          gmsh::model::mesh::field::setNumbers(abs_distance_fid, "CurvesList",
                                               boundary_tags);
          break;
        case 2:
          gmsh::model::mesh::field::setNumbers(abs_distance_fid, "SurfacesList",
                                               boundary_tags);
          break;
        default:
          LOG_ERROR("Invalid dimension");
          return -1;
        } // dim switch

        std::string const abs_distance_name = 'F' + std::to_string(abs_distance_fid);

        // Create a field that multiplies the base field by the distance to the abs
        // material boundary
        int const fid = gmsh::model::mesh::field::add("MathEval");
        double const scale = lcs[i] * sigmas_t[i] * abs_mfp_scale;
        double const offset = lcs[i] * (1.0 - abs_mfp_threshold * abs_mfp_scale);
        std::string math_expr = abs_distance_name + " * " + std::to_string(scale);
        if (offset >= 0.0) {
          math_expr += " + " + std::to_string(offset);
        } else {
          math_expr += " - " + std::to_string(-offset);
        }
        LOG_INFO("Creating linear field for ", materials[static_cast<Int>(i)].getName(),
                 ": ", math_expr.c_str());
        gmsh::model::mesh::field::setString(fid, "F", math_expr);
        int const max_fid = gmsh::model::mesh::field::add("Max");
        std::vector<double> const field_ids_d = {static_cast<double>(field_ids[i]),
                                                 static_cast<double>(fid)};
        gmsh::model::mesh::field::setNumbers(max_fid, "FieldsList", field_ids_d);
        // Replace the base field with the max field
        field_ids[i] = max_fid;
      } // if (is_absorber[i] == 1)
    } // for (size_t i = 0; i < num_materials; ++i)
  } // if (abs_mfp_threshold > 0.0)

  // Create a field that takes the min of each and set as background mesh
  int const fid = gmsh::model::mesh::field::add("Min");
  std::vector<double> const double_field_ids(field_ids.begin(), field_ids.end());
  gmsh::model::mesh::field::setNumbers(fid, "FieldsList", double_field_ids);
  gmsh::model::mesh::field::setAsBackgroundMesh(fid);
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", 0);

  return 0;
}

} // namespace um2::gmsh::model::mesh
#endif // UM2_USE_GMSH
