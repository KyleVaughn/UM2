#if UM2_ENABLE_GMSH

#  include <iomanip>

namespace um2::gmsh::model::occ
{

template <std::floating_point T, std::signed_integral I>
void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
overlaySpatialPartition(mpact::SpatialPartition<T, I> const & partition,
                        std::string const & fill_material_name,
                        Color const fill_material_color)
{
  Log::info("Overlaying MPACT spatial partition");
  // Algorithm:
  //  1. Get the lower left corner of each unique 2D coarse cell rectangle that
  //      will cut the model.
  //  2. Create the rectangles. Since group preserving fragment only keeps the
  //      highest dimension groups, if the model is 3D, we create a box whose
  //      bottom face is the coarse cell rectangle.
  //  3. Assign the fill material and coarse cell labels to these entities.
  //  4. Group preserving fragment with the model and grid.
  //  4a. If the model is 3D, group preserving intersection between the model
  //      and the 2D rectangles.
  //  5. Remove all entities that are not coarse cells.

  // Get all model entities prior to adding the grid
  gmsh::vectorpair model_dimtags;
  gmsh::model::getEntities(model_dimtags, 3);
  // Determine model dimension
  int model_dim = 3;
  if (model_dimtags.empty()) {
    gmsh::model::getEntities(model_dimtags, 2);
    model_dim = 2;
  }
  // Get the unique coarse cell lower left corners
  Size const num_cc = partition.numCoarseCells();
  Vector<Point3<T>> cc_lower_lefts(num_cc); // Of the cut-plane
  Vector<Vec3<T>> cc_extents(num_cc);
  Vector<int8_t> cc_found(num_cc, 0);
  Vector<int8_t> rtm_found(partition.numRTMs(), 0);
  Vector<int8_t> lat_found(partition.numLattices(), 0);
  Vector<int8_t> asy_found(partition.numAssemblies(), 0);

  auto const & core = partition.core;
  if (core.children.empty()) {
    Log::error("Core has no children");
    return;
  }
  // For each assembly
  Size const nyasy = core.numYCells();
  Size const nxasy = core.numXCells();
  for (Size iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Size ixasy = 0; ixasy < nxasy; ++ixasy) {
      auto const asy_id = static_cast<Size>(core.getChild(ixasy, iyasy));
      if (asy_found[asy_id] == 1) {
        continue;
      }
      asy_found[asy_id] = 1;
      AxisAlignedBox2<T> const asy_bb = core.getBox(ixasy, iyasy);
      Point2<T> const asy_ll = asy_bb.minima; // Lower left corner
      auto const & assembly = partition.assemblies[asy_id];
      if (assembly.children.empty()) {
        Log::error("Assembly has no children");
        return;
      }
      // For each lattice
      Size const nzlat = assembly.numXCells();
      for (Size izlat = 0; izlat < nzlat; ++izlat) {
        auto const lat_id = static_cast<Size>(assembly.getChild(izlat));
        if (lat_found[lat_id] == 1) {
          continue;
        }
        lat_found[lat_id] = 1;
        T const low_z = assembly.grid.divs[0][izlat];
        T const high_z = assembly.grid.divs[0][izlat + 1];
        T const z_cut = (low_z + high_z) / 2;
        T const dz =
            high_z - low_z; // This was changed in the refactor. Used to be divided by 2.
        auto const & lattice = partition.lattices[lat_id];
        if (lattice.children.empty()) {
          Log::error("Lattice has no children");
          return;
        }
        // For each RTM
        Size const nyrtm = lattice.numYCells();
        Size const nxrtm = lattice.numXCells();
        for (Size iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (Size ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {
            auto const rtm_id = static_cast<Size>(lattice.getChild(ixrtm, iyrtm));
            if (rtm_found[rtm_id] == 1) {
              continue;
            }
            rtm_found[rtm_id] = 1;
            AxisAlignedBox2<T> const rtm_bb = lattice.getBox(ixrtm, iyrtm);
            Point2<T> const rtm_ll = rtm_bb.minima; // Lower left corner
            auto const & rtm = partition.rtms[rtm_id];
            if (rtm.children.empty()) {
              Log::error("RTM has no children");
              return;
            }
            // For each coarse cell
            Size const nycells = rtm.numYCells();
            Size const nxcells = rtm.numXCells();
            for (Size iycell = 0; iycell < nycells; ++iycell) {
              for (Size ixcell = 0; ixcell < nxcells; ++ixcell) {
                auto const cell_id = static_cast<Size>(rtm.getChild(ixcell, iycell));
                if (cc_found[cell_id] == 1) {
                  continue;
                }
                cc_found[cell_id] = 1;
                AxisAlignedBox2<T> const cell_bb = rtm.getBox(ixcell, iycell);
                Point2<T> const ll = asy_ll + rtm_ll + cell_bb.minima;
                cc_lower_lefts[cell_id] = {ll[0], ll[1], z_cut};
                cc_extents[cell_id] = {cell_bb.width(), cell_bb.height(), dz};
              } // cell
            }   // cell
          }     // rtm
        }       // rtm
      }         // lat
    }           // assembly
  }             // assembly

  // Get materials and see if the fill material already exists
  // If it does, move it to the end of the material hierarchy, otherwise
  // append it to the end.
  std::vector<Material> materials;
  um2::gmsh::model::getMaterials(materials);
  bool fill_exists = false;
  size_t const num_materials = materials.size();
  for (size_t i = 0; i < num_materials; ++i) {
    std::string const name(materials[i].name.data());
    if (name == fill_material_name) {
      // Move the material to the end of the list to ensure
      // it is the fill material
      std::swap(materials[i], materials[num_materials - 1]);
      fill_exists = true;
      Log::debug("Found fill material: " + fill_material_name);
      break;
    }
  }
  if (!fill_exists) {
    materials.emplace_back(ShortString(fill_material_name.c_str()), fill_material_color);
  }
  std::vector<int> cc_tags(static_cast<size_t>(num_cc));
  namespace factory = gmsh::model::occ;
  if (model_dim == 2) {
    // Create rectangles
    for (Size i = 0; i < num_cc; ++i) {
      Point3<T> const & ll = cc_lower_lefts[i];
      Vec3<T> const & ext = cc_extents[i];
      cc_tags[static_cast<size_t>(i)] =
          factory::addRectangle(ll[0], ll[1], ll[2], ext[0], ext[1]);
    }
  } else {
    // Create boxes
    for (Size i = 0; i < num_cc; ++i) {
      Point3<T> const & ll = cc_lower_lefts[i];
      Vec3<T> const & ext = cc_extents[i];
      cc_tags[static_cast<size_t>(i)] =
          factory::addBox(ll[0], ll[1], ll[2], ext[0], ext[1], ext[2]);
    }
  }
  factory::synchronize();

  // Assign physical groups
  // Fill material is assigned to all slice rectangles
  std::string const fill_mat_full_name = "Material " + fill_material_name;
  addToPhysicalGroup(model_dim, cc_tags, -1, fill_mat_full_name);
  // Add a physical group for each coarse cell
  std::stringstream ss;
  for (Size i = 0; i < num_cc; ++i) {
    ss.str("");
    ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << i;
    gmsh::model::addPhysicalGroup(model_dim, {cc_tags[static_cast<size_t>(i)]}, -1,
                                  ss.str());
  }
  // Fragment
  gmsh::vectorpair grid_dimtags(cc_tags.size());
  for (size_t i = 0; i < cc_tags.size(); ++i) {
    grid_dimtags[i] = {model_dim, cc_tags[i]};
  }
  std::sort(grid_dimtags.begin(), grid_dimtags.end());
  gmsh::vectorpair out_dimtags;
  std::vector<gmsh::vectorpair> out_dimtags_map;
  groupPreservingFragment(model_dimtags, grid_dimtags, out_dimtags, out_dimtags_map,
                          materials);

  // Remove all entities that do not have a "Coarse_Cell" physical group
  // associated with them.
  std::vector<int> dont_remove;
  gmsh::vectorpair group_dimtags;
  gmsh::model::getPhysicalGroups(group_dimtags, model_dim);
  for (auto const & dimtag : group_dimtags) {
    int const dim = dimtag.first;
    assert(dim == model_dim);
    int const tag = dimtag.second;
    std::string name;
    // We do not expect entities to have many physical groups. Therefore,
    // we insert the whole list of entities for each physical group into
    // the to_remove vector.
    gmsh::model::getPhysicalName(dim, tag, name);
    if (name.starts_with("Coarse_Cell")) {
      std::vector<int> tags;
      gmsh::model::getEntitiesForPhysicalGroup(dim, tag, tags);
      dont_remove.insert(dont_remove.end(), tags.begin(), tags.end());
    }
  }
  std::sort(dont_remove.begin(), dont_remove.end());
  gmsh::vectorpair dont_remove_dimtags;
  if (!dont_remove.empty()) {
    dont_remove_dimtags.emplace_back(model_dim, dont_remove[0]);
  }
  for (size_t i = 1; i < dont_remove.size(); ++i) {
    if (dont_remove[i] != dont_remove[i - 1]) {
      dont_remove_dimtags.emplace_back(model_dim, dont_remove[i]);
    }
  }
  Log::info("Removing unneeded entities");
  gmsh::vectorpair to_remove_dimtags;
  gmsh::vectorpair all_dimtags;
  gmsh::model::getEntities(
      all_dimtags, model_dim); // This was changed in the refactor to include model_dim
  std::set_difference(all_dimtags.begin(), all_dimtags.end(), dont_remove_dimtags.begin(),
                      dont_remove_dimtags.end(), std::back_inserter(to_remove_dimtags));
  gmsh::model::occ::remove(to_remove_dimtags, /*recursive=*/true);
  gmsh::model::removeEntities(to_remove_dimtags, /*recursive=*/true);
  gmsh::model::occ::synchronize();
  if (model_dim == 3) {
    Log::error("Tell Kyle to reimplement 3D");
  }
  //    if (model_dim == 3) {
  //        // We need to create a 2D version of each physical group, then intersect the
  //        2D
  //        // grid.
  //        group_dimtags.clear();
  //        gmsh::model::getPhysicalGroups(group_dimtags);
  //        for (auto const & dimtag : group_dimtags) {
  //            int const dim = dimtag.first;
  //            UM2_ASSERT(dim == 3);
  //            int const tag = dimtag.second;
  //            std::string name;
  //            gmsh::model::getPhysicalName(dim, tag, name);
  //            std::vector<int> tags_3d;
  //            gmsh::model::getEntitiesForPhysicalGroup(dim, tag, tags_3d);
  //            std::vector<int> tags_2d;
  //            for (auto const & etag : tags_3d) {
  //                std::vector<int> upward, downward;
  //                gmsh::model::getAdjacencies(3, etag, upward, downward);
  //                // If the downard tags are not in the list of 2D entities, then add
  //                them.
  //                // We expect many repeated tags, so we use a binary search to find the
  //                // insertion point.
  //                for (auto const & tag_2 : downward) {
  //                    auto it = std::lower_bound(tags_2d.begin(), tags_2d.end(), tag_2);
  //                    if (it == tags_2d.end() || *it != tag_2) {
  //                        tags_2d.insert(it, tag_2);
  //                    }
  //                }
  //            }
  //            // Create a new physical group for the 2D entities
  //            gmsh::model::addPhysicalGroup(2, tags_2d, -1, name);
  //        }
  //        gmsh::vectorpair model_dimtags_2d;
  //        gmsh::model::getEntities(model_dimtags_2d, 2);
  //        std::vector<int> cc_tags_2d(num_cc);
  //        // Create rectangles
  //        for (Size i = 0; i < num_cc; ++i) {
  //            Point3<T> const & ll = cc_lower_lefts[i];
  //            Vec3<T> const & ext = cc_extents[i];
  //            cc_tags_2d[static_cast<size_t>(i)] =
  //                factory::addRectangle(ll[0], ll[1], ll[2], ext[0], ext[1]);
  //        }
  //        factory::synchronize();
  //        // Don't need to add coarse cell physical groups to the 2D grid. The model
  //        // already has the physical groups.
  //        gmsh::vectorpair grid_dimtags_2d(cc_tags_2d.size());
  //        for (size_t i = 0; i < cc_tags_2d.size(); ++i) {
  //            grid_dimtags_2d[i] = {2, cc_tags_2d[i]};
  //        }
  //        std::sort(grid_dimtags_2d.begin(), grid_dimtags_2d.end());
  //        // remove all 3D entities (leaving surfaces), then intersect the 2D grid
  //        to_remove_dimtags.clear();
  //        gmsh::model::getEntities(to_remove_dimtags, 3);
  //        gmsh::model::removeEntities(to_remove_dimtags);
  //        gmsh::model::occ::remove(to_remove_dimtags);
  //        gmsh::vectorpair out_dimtags_2d;
  //        std::vector<gmsh::vectorpair> out_dimtags_map_2d;
  //        group_preserving_intersect(model_dimtags_2d, grid_dimtags_2d,
  //                out_dimtags_2d, out_dimtags_map_2d, materials);
  //        // Remove all entities that are not children of the grid entities.
  //        std::vector<int> dont_remove_2d;
  //        size_t const nmodel_2d = model_dimtags_2d.size();
  //        size_t const ngrid_2d = grid_dimtags_2d.size();
  //        for (size_t i = nmodel_2d; i < nmodel_2d + ngrid_2d; ++i) {
  //            for (auto const & child : out_dimtags_map_2d[i]) {
  //                auto child_it = std::lower_bound(dont_remove_2d.begin(),
  //                                                 dont_remove_2d.end(),
  //                                                 child.second);
  //                if (child_it == dont_remove_2d.end() || *child_it != child.second) {
  //                    dont_remove_2d.insert(child_it, child.second);
  //                }
  //            }
  //        }
  //        gmsh::vectorpair dont_remove_dimtags_2d(dont_remove_2d.size());
  //        for (size_t i = 0; i < dont_remove_2d.size(); ++i) {
  //            dont_remove_dimtags_2d[i] = {2, dont_remove_2d[i]};
  //        }
  //        gmsh::vectorpair to_remove_dimtags_2d;
  //        gmsh::vectorpair all_dimtags_2d;
  //        gmsh::model::getEntities(all_dimtags_2d, 2);
  //        std::set_difference(all_dimtags_2d.begin(), all_dimtags_2d.end(),
  //                dont_remove_dimtags_2d.begin(), dont_remove_dimtags_2d.end(),
  //                std::back_inserter(to_remove_dimtags_2d));
  //        gmsh::vectorpair all_dimtags_3d;
  //        gmsh::model::getEntities(all_dimtags_3d, 3);
  //        gmsh::model::occ::remove(all_dimtags_3d);
  //        gmsh::model::removeEntities(all_dimtags_3d);
  ////        gmsh::model::occ::remove(to_remove_dimtags_2d, true); Already removed via
  /// intersect
  //        gmsh::model::removeEntities(to_remove_dimtags_2d, true);
  //        gmsh::model::occ::synchronize();
  //    }
}

} // namespace um2::gmsh::model::occ
#endif // UM2_ENABLE_GMSH
