#include <um2/common/cast_if_not.hpp>
#include <um2/common/logger.hpp>
#include <um2/config.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/polytope.hpp>
#include <um2/math/stats.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/mesh/polytope_soup.hpp>
#include <um2/mpact/powers.hpp>
#include <um2/stdlib/algorithm/copy.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/pair.hpp>
#include <um2/stdlib/vector.hpp>

// NOLINTBEGIN(misc-include-cleaner)
#include <um2/geometry/quadratic_quadrilateral.hpp>
#include <um2/geometry/quadratic_triangle.hpp>
#include <um2/geometry/quadrilateral.hpp>
#include <um2/geometry/triangle.hpp>
// NOLINTEND(misc-include-cleaner)

#include <algorithm>

namespace um2::mpact
{

namespace
{

void
getPowerData(PolytopeSoup const & soup, Vector<Int> & ids, Vector<Float> & power_data)
{
  soup.getElset("power", ids, power_data);
  if (ids.size() != power_data.size()) {
    LOG_ERROR("Mismatch in number of ids and data");
  }
  if (ids.empty()) {
    LOG_ERROR("No power data found");
  }
  if (ids.size() != soup.numElements()) {
    LOG_ERROR("Mismatch in number of ids and elements");
  }
  if (!um2::is_sorted(ids.cbegin(), ids.cend())) {
    LOG_ERROR("IDs are not sorted");
  }
  // Ensure there is at least one non-zero power
  for (auto const p : power_data) {
    if (p > 0) {
      return;
    }
  }
  LOG_ERROR("No non-zero power FSRs found");
}

template <Int P, Int N>
PURE [[nodiscard]] auto
facesShareVertex(PlanarPolygon<P, N, Float> const & a,
                 PlanarPolygon<P, N, Float> const & b) -> bool
{
  auto constexpr point_tolerance = castIfNot<Float>(1e-5);
  for (Int i = 0; i < N; ++i) {
    for (Int j = 0; j < N; ++j) {
      if (a[i].distanceTo(b[j]) < point_tolerance) {
        return true;
      }
    }
  }
  return false;
}

// Specialize for each mesh type
template <Int P, Int N>
[[nodiscard]] auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
getPowers(PolytopeSoup const & soup) -> Vector<Pair<Float, Point2F>>
{
  // Get the power data
  //---------------------------------------------------------------------------
  Vector<Int> ids;
  Vector<Float> power_data;
  getPowerData(soup, ids, power_data);

  // Create the FVM, then compute the AABB of each element which has non-zero
  // power. We will store the indices of these elements in a separate vector.
  //--------------------------------------------------------------------------
  // We turn off validation, since the mesh will likely not be watertight
  FaceVertexMesh<P, N> const fvm(soup, /*validate=*/false);
  Int const num_faces = fvm.numFaces();
  Vector<Int> nonzero_power_ids;
  nonzero_power_ids.reserve(num_faces);
  // We allocate AABBs for each face, but only populate the ones that have
  // non-zero power. This makes indexing easier.
  Vector<AxisAlignedBox2F> face_aabbs(num_faces);
  for (Int i = 0; i < num_faces; ++i) {
    if (power_data[i] > 0) {
      nonzero_power_ids.emplace_back(i);
      face_aabbs[i] = fvm.getFace(i).boundingBox();
      // Scale the box up by 1% to avoid floating point issues
      // with AABB intersection tests later on
      auto constexpr scale = castIfNot<Float>(1.01);
      face_aabbs[i].scale(scale);
      ASSERT(ids[i] == i);
    }
  }

  // Do an initial pass through the data to find connected subsets of faces
  // that have non-zero power. This will not find all subsets due to the
  // order in which the faces are processed. Hence, we will merge the subsets
  // next.
  //
  // We will keep track of the IDs of the faces in each subset, as well as the
  // AABB of the subset.
  //--------------------------------------------------------------------------
  Vector<Vector<Int>> subset_ids;
  Vector<AxisAlignedBox2F> subset_aabbs;
  for (auto const iface : nonzero_power_ids) {
    bool added_to_subset = false;
    auto const & face_aabb = face_aabbs[iface];
    for (Int iset = 0; iset < subset_ids.size(); ++iset) {
      auto & subset = subset_ids[iset];
      auto & subset_aabb = subset_aabbs[iset];
      // If the bounding box of the subset intersects the bounding box of the
      // current face, then the current face may belong in the subset
      if (face_aabb.intersects(subset_aabb)) {
        // Check to see if the current face shares a vertex with any face in
        // the subset. We do a fuzzy check since the mesh may not be watertight.
        auto const this_face = fvm.getFace(iface);
        for (auto const subset_face_id : subset) {
          auto const subset_face = fvm.getFace(subset_face_id);
          if (facesShareVertex(this_face, subset_face)) {
            subset.emplace_back(iface);
            subset_aabb += face_aabb;
            added_to_subset = true;
            goto next_face;
          }
        } // for subset_face_id
      } // if intersects
    } // for iset
  next_face:
    if (!added_to_subset) {
      // New subset
      Vector<Int> new_subset(1);
      new_subset[0] = iface;
      subset_ids.emplace_back(um2::move(new_subset));
      subset_aabbs.emplace_back(face_aabb);
    } // if !added_to_subset
  } // for iface

  // We will now merge adjacent subsets. We keep iterating until no more merges
  // are possible.
  // Algorithm:
  //  Suppose we have N subsets.
  //  for i in [N - 1, N - 2, ..., 1]
  //    for j in [i - 1, i - 2, ..., 0]
  //      if aabb_j intersects aabb_i
  //        if any face in subset j shares a vertex with any face in subset i
  //          merge i into j
  //          remove subset i (This decrements N by 1)
  //          restart the outer loop (i)
  //--------------------------------------------------------------------------
  bool done_merging = false;
  Int merge_count = 0;
  while (!done_merging) {
    done_merging = true;
    for (Int i = subset_ids.size() - 1; i > 0; --i) {
      auto const & subset_i = subset_ids[i];
      auto const & aabb_i = subset_aabbs[i];
      for (Int j = i - 1; j >= 0; --j) {
        auto const & subset_j = subset_ids[j];
        auto const & aabb_j = subset_aabbs[j];
        if (aabb_j.intersects(aabb_i)) {
          for (auto const id_i : subset_i) {
            auto const face_i = fvm.getFace(id_i);
            for (auto const id_j : subset_j) {
              auto const face_j = fvm.getFace(id_j);
              if (facesShareVertex(face_i, face_j)) {
                Int const set_i_size = subset_ids[i].size();
                Int const set_j_size = subset_ids[j].size();
                subset_ids[j].resize(set_i_size + set_j_size);
                um2::copy(subset_ids[i].cbegin(), subset_ids[i].cend(),
                          subset_ids[j].begin() + set_j_size);
                std::sort(subset_ids[j].begin(), subset_ids[j].end());
                subset_aabbs[j] += subset_aabbs[i];
                // We will not erase the subset i, by moving all the subsets
                // with a higher index down by one.
                for (Int k = i; k < subset_ids.size() - 1; ++k) {
                  subset_ids[k] = um2::move(subset_ids[k + 1]);
                  subset_aabbs[k] = subset_aabbs[k + 1];
                }
                subset_ids.pop_back();
                subset_aabbs.pop_back();
                done_merging = false;
                goto next_merge;
              }
            } // for id_j
          } // for id_i
        } // if intersects
      } // for j
    } // for i
  next_merge:
    ++merge_count;
  } // while !done_merging
  LOG_INFO("Merging of FSR subsets complete after ", merge_count, " iterations");

  // Subset power and centroid vector
  Vector<Pair<Float, Point2F>> subset_pc(subset_ids.size());
  Vector<Float> areas;
  Vector<Point2F> area_weighted_centroids;
  Vector<Float> area_weighted_powers;
  for (Int iset = 0; iset < subset_ids.size(); ++iset) {
    auto const & subset = subset_ids[iset];
    Int const n = subset.size();
    areas.resize(n);
    area_weighted_centroids.resize(n);
    area_weighted_powers.resize(n);
    // Get the area, area-weighted centroid, and area-weighted power
    for (Int i = 0; i < n; ++i) {
      Int const iface = subset[i];
      auto const face = fvm.getFace(iface);
      Float const a = face.area();
      Point2F const c = face.centroid();
      areas[i] = a;
      area_weighted_centroids[i] = a * c;
      area_weighted_powers[i] = power_data[iface] * a;
      ASSERT(power_data[iface] > 0);
    }
    Float const area_sum = um2::sum(areas.cbegin(), areas.cend());
    Point2F centroid_sum =
        um2::sum(area_weighted_centroids.cbegin(), area_weighted_centroids.cend());
    Float const total_power =
        um2::sum(area_weighted_powers.cbegin(), area_weighted_powers.cend());

    // Compute the centroid of the set from geometric decomposition
    // c = sum_i (a_i * c_i) / sum_i a_i
    centroid_sum /= area_sum;
    subset_pc[iset] = Pair(total_power, centroid_sum);
  }
  return subset_pc;
}

} // namespace

PURE [[nodiscard]] auto
getPowers(String const & fsr_output) -> Vector<Pair<Float, Point2F>>
{
  LOG_INFO("Extracting power and centroid from MPACT FSR output");
  PolytopeSoup const soup(fsr_output);

  // For now, we assume that all the elements are the same type.
  auto const elem_types = soup.getElemTypes();
  if (elem_types.size() != 1) {
    LOG_ERROR("Expected only one element type, but found ", elem_types.size());
    return {};
  }
  switch (elem_types[0]) {
  case VTKElemType::Triangle:
    LOG_INFO("FSR mesh type: Triangle");
    return getPowers<1, 3>(soup);
  case VTKElemType::Quad:
    LOG_INFO("FSR mesh type: Quad");
    return getPowers<1, 4>(soup);
  case VTKElemType::QuadraticTriangle:
    LOG_INFO("FSR mesh type: QuadraticTriangle");
    return getPowers<2, 6>(soup);
  case VTKElemType::QuadraticQuad:
    LOG_INFO("FSR mesh type: QuadraticQuad");
    return getPowers<2, 8>(soup);
  default:
    LOG_ERROR("Unsupported element type");
    return {};
  }
}

} // namespace um2::mpact
