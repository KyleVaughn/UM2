#include <um2/common/cast_if_not.hpp>
#include <um2/common/logger.hpp>
#include <um2/common/strto.hpp>
#include <um2/config.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>
#include <um2/math/matrix.hpp>
#include <um2/math/stats.hpp>
#include <um2/math/vec.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/mesh/polytope_soup.hpp>
#include <um2/mesh/rectilinear_grid.hpp>
#include <um2/mesh/rectilinear_partition.hpp>
#include <um2/mesh/regular_grid.hpp>
#include <um2/mesh/regular_partition.hpp>
#include <um2/mpact/model.hpp>
#include <um2/physics/cmfd.hpp>
#include <um2/physics/material.hpp>
#include <um2/stdlib/algorithm/copy.hpp>
#include <um2/stdlib/algorithm/fill.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>
#include <um2/stdlib/algorithm/min.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/math/roots.hpp>
#include <um2/stdlib/math/trigonometric_functions.hpp>
#include <um2/stdlib/numbers.hpp>
#include <um2/stdlib/numeric/iota.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/string_view.hpp>
#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/pair.hpp>
#include <um2/stdlib/vector.hpp>

#if UM2_USE_PUGIXML
#  include <pugixml.hpp>
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>

// We dont have access to the header defining many of the HDF5 types, so we disable
// the clang-tidy warning.
// NOLINTBEGIN(misc-include-cleaner)

namespace um2::mpact
{

//=============================================================================
// flattenLattice
//=============================================================================
// Helper function to convert a 2D vector of integers into a 1D vector of integers
//  um2::Vector<um2::Vector<Int>> const ids = {
//      {2, 3},
//      {0, 1}
//  };
//
//  -> um2::Vector<Int> flat_ids = {0, 1, 2, 3};

namespace
{

template <typename T, typename U>
void
flattenLattice(Vector<Vector<T>> const & ids, Vector<U> & flat_ids)
{
  Int const num_rows = ids.size();
  Int const num_cols = ids[0].size();
  // Ensure all rows have the same number of columns
  for (Int i = 1; i < num_rows; ++i) {
    if (ids[i].size() != num_cols) {
      logger::error("Each row must have the same number of columns");
      return;
    }
  }
  flat_ids.resize(num_rows * num_cols);
  for (Int i = 0; i < num_rows; ++i) {
    for (Int j = 0; j < num_cols; ++j) {
      flat_ids[i * num_cols + j] = static_cast<T>(ids[num_rows - 1 - i][j]);
    }
  }
}

} // namespace

//=============================================================================
// Constructors
//=============================================================================

Model::Model(String const & filename) { read(filename); }

//=============================================================================
// clear
//=============================================================================

HOSTDEV void
Model::clear() noexcept
{
  _core.clear();
  _assemblies.clear();
  _lattices.clear();
  _rtms.clear();
  _coarse_cells.clear();

  _materials.clear();

  _tris.clear();
  _quads.clear();
  _tri6s.clear();
  _quad8s.clear();
}

//=============================================================================
// addMaterial
//=============================================================================

auto
Model::addMaterial(Material const & material, bool const validate) -> Int
{
  if (validate) {
    material.validateXSec();
  }
  _materials.emplace_back(material);
  return _materials.size() - 1;
}

//=============================================================================
// addCylindricalPinMesh
//=============================================================================

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::addCylindricalPinMesh(Float const pitch, Vector<Float> const & radii,
                             Vector<Int> const & num_rings, Int const num_azimuthal,
                             Int const mesh_order) -> Int
{

  Int mesh_id = -1;
  if (mesh_order == 1) {
    mesh_id = this->_quads.size();
    LOG_DEBUG("Adding quad cylindrical pin mesh ", mesh_id);
  } else if (mesh_order == 2) {
    mesh_id = this->_quad8s.size();
    LOG_DEBUG("Adding quad8 cylindrical pin mesh ", mesh_id);
  } else {
    logger::error("Invalid mesh order");
    return -1;
  }

  if (num_azimuthal < 8) {
    logger::error("The number of azimuthal divisions must be at least 8");
    return -1;
  }

  if ((num_azimuthal & (num_azimuthal - 1)) != 0) {
    logger::error("The number of azimuthal divisions must be a power of 2");
    return -1;
  }

  if (radii.size() != num_rings.size()) {
    logger::error("The suze of radii must match the size of num_rings");
    return -1;
  }

  if (std::any_of(radii.begin(), radii.end(),
                  [pitch](Float r) { return r > pitch / 2; })) {
    logger::error("The radii must be less than half the pitch");
    return -1;
  }

  auto constexpr eps = epsDistance<Float>();
#if UM2_ENABLE_ASSERTS
  Float constexpr big_eps = 100 * eps;
#endif

  // radial region = region containing different materials (rings + outside of
  // the last radius)
  //
  // radial_region_areas = area of each radial region, including outside of the last
  // radius
  //
  // ring = an equal area division of a radial region containing the same material
  //
  // ring_radii = the radius of each ring, NOF including the outside of the last
  // radius
  //
  // ring_areas = the area of each ring, including the outside of the last radius

  //---------------------------------------------------------------------------
  // Get the area of each radial region (rings + outside of the last ring)
  //---------------------------------------------------------------------------
  Int const num_radial_regions = radii.size() + 1;
  Vector<Float> radial_region_areas(num_radial_regions);
  // A0 = pi * r0^2
  // Ai = pi * (ri^2 - ri-1^2)
  radial_region_areas[0] = pi<Float> * radii[0] * radii[0];
  for (Int i = 1; i < num_radial_regions - 1; ++i) {
    radial_region_areas[i] =
        pi<Float> * (radii[i] * radii[i] - radii[i - 1] * radii[i - 1]);
  }
  radial_region_areas[num_radial_regions - 1] =
      pitch * pitch - radial_region_areas[num_radial_regions - 2];

  //---------------------------------------------------------------------------
  // Get the radii and areas of each ring after splitting the radial regions
  // This includes outside of the last ring
  //---------------------------------------------------------------------------
  Int const total_rings = std::reduce(num_rings.begin(), num_rings.end(), 0);
  Vector<Float> ring_radii(total_rings);
  Vector<Float> ring_areas(total_rings + 1);
  // Inside the innermost region
  ring_areas[0] = radial_region_areas[0] / static_cast<Float>(num_rings[0]);
  ring_radii[0] = um2::sqrt(ring_areas[0] / pi<Float>);
  for (Int i = 1; i < num_rings[0]; ++i) {
    ring_areas[i] = ring_areas[0];
    ring_radii[i] =
        um2::sqrt(ring_areas[i] / pi<Float> + ring_radii[i - 1] * ring_radii[i - 1]);
  }
  Int ctr = num_rings[0];
  for (Int ireg = 1; ireg < num_radial_regions - 1; ++ireg) {
    Int const num_rings_in_region = num_rings[ireg];
    Float const area_per_ring =
        radial_region_areas[ireg] / static_cast<Float>(num_rings_in_region);
    for (Int iring = 0; iring < num_rings_in_region; ++iring, ++ctr) {
      ring_areas[ctr] = area_per_ring;
      ring_radii[ctr] = um2::sqrt(area_per_ring / pi<Float> +
                                  ring_radii[ctr - 1] * ring_radii[ctr - 1]);
    }
  }
  // Outside of the last ring
  ring_areas[ctr] = pitch * pitch - pi<Float> * ring_radii.back() * ring_radii.back();
  // Log the radii and areas in debug mode
  for (Int i = 0; i < total_rings; ++i) {
    LOG_DEBUG("Ring ", i, " radius: ", ring_radii[i]);
    LOG_DEBUG("Ring ", i, " area: ", ring_areas[i]);
  }
  LOG_DEBUG("The area outside of the last ring is ", ring_areas[ctr]);
#if UM2_ENABLE_ASSERTS
  // Ensure the sum of the ring areas is equal to pitch^2
  Float const sum_ring_areas = std::reduce(ring_areas.begin(), ring_areas.end());
  ASSERT_NEAR(sum_ring_areas, pitch * pitch, eps);
#endif
  auto const num_azimuthal_t = static_cast<Float>(num_azimuthal);
  if (mesh_order == 1) {
    // Get the equivalent radius of each ring if it were a quadrilateral
    Float const theta = 2 * pi<Float> / num_azimuthal_t;
    Float const sin_theta = um2::sin(theta);
    Vector<Float> eq_radii(total_rings);
    // The innermost radius is a special case, and is essentially a triangle.
    // A_t = l² * sin(θ) / 2
    // A_ring = num_azi * A_t = l² * sin(θ) * num_azi / 2
    // l = sqrt(2 * A_ring / (sin(θ) * num_azi))
    eq_radii[0] = um2::sqrt(2 * ring_areas[0] / (sin_theta * num_azimuthal_t));
    // A_q = (l² - l²₀) * sin(θ) / 2
    // A_ring = num_azi * A_q = (l² - l²₀) * sin(θ) * num_azi / 2
    // l = sqrt(2 * A_ring / (sin(θ) * num_azi) + l²₀)
    for (Int i = 1; i < total_rings; ++i) {
      eq_radii[i] = um2::sqrt(2 * ring_areas[i] / (sin_theta * num_azimuthal_t) +
                              eq_radii[i - 1] * eq_radii[i - 1]);
    }
    for (Int i = 0; i < total_rings; ++i) {
      LOG_DEBUG("Ring ", i, " equivalent radius: ", eq_radii[i]);
    }
    // If any of the equivalent radii are larger than half the pitch, error
    if (std::any_of(eq_radii.begin(), eq_radii.end(),
                    [pitch](Float r) { return r > pitch / 2; })) {
      logger::error("The equivalent radius of a ring is larger than half the pitch");
      return -1;
    }
    // Sanity check: ensure the sum of the quadrilateral areas in a ring is equal to
    // the ring area
#if UM2_ENABLE_ASSERTS
    ASSERT_NEAR(eq_radii[0] * eq_radii[0] * sin_theta / 2,
                ring_areas[0] / num_azimuthal_t, big_eps);
    for (Int i = 1; i < total_rings; ++i) {
      Float const area =
          (eq_radii[i] * eq_radii[i] - eq_radii[i - 1] * eq_radii[i - 1]) * sin_theta / 2;
      ASSERT_NEAR(area, ring_areas[i] / num_azimuthal_t, big_eps);
    }
#endif

    //------------------------------------------------------------------------
    // Get the points that make up the mesh
    //------------------------------------------------------------------------
    // The number of points is:
    //   Center point
    //   num_azimuthal / 2, for the points in the innermost ring to make the quads
    //      "triangular"
    //   (num_rings + 1) * num_azimuthal
    Int const num_points = 1 + (total_rings + 1) * num_azimuthal + num_azimuthal / 2;
    Vector<Point2F> vertices(num_points);
    LOG_DEBUG("The number of points is ", num_points);
    // Center point
    vertices[0] = {0, 0};
    // Triangular points
    LOG_DEBUG("Computing the triangular points");
    Float const rt = eq_radii[0] / 2;
    for (Int ia = 0; ia < num_azimuthal / 2; ++ia) {
      Float const sin_ia_theta = um2::sin(theta * (2 * static_cast<Float>(ia) + 1));
      Float const cos_ia_theta = um2::cos(theta * (2 * static_cast<Float>(ia) + 1));
      vertices[1 + ia] = {rt * cos_ia_theta, rt * sin_ia_theta};
    }
    LOG_DEBUG("Computing the quadrilateral points");
    // Quadrilateral points
    // Points on rings, not including the boundary of the pin (pitch / 2 box)
    for (Int ir = 0; ir < total_rings; ++ir) {
      Int const num_prev_points = 1 + num_azimuthal / 2 + ir * num_azimuthal;
      for (Int ia = 0; ia < num_azimuthal; ++ia) {
        Float sin_ia_theta = um2::sin(theta * static_cast<Float>(ia));
        Float cos_ia_theta = um2::cos(theta * static_cast<Float>(ia));
        if (um2::abs(sin_ia_theta) < eps) {
          sin_ia_theta = 0;
        }
        if (um2::abs(cos_ia_theta) < eps) {
          cos_ia_theta = 0;
        }
        vertices[num_prev_points + ia] = {eq_radii[ir] * cos_ia_theta,
                                          eq_radii[ir] * sin_ia_theta};
      }
    }
    LOG_DEBUG("Computing the boundary points");
    // Points on the boundary of the pin (pitch / 2)
    Int const num_prev_points = 1 + num_azimuthal / 2 + total_rings * num_azimuthal;
    for (Int ia = 0; ia < num_azimuthal; ++ia) {
      Float sin_ia_theta = um2::sin(theta * static_cast<Float>(ia));
      Float cos_ia_theta = um2::cos(theta * static_cast<Float>(ia));
      if (um2::abs(sin_ia_theta) < eps) {
        sin_ia_theta = 0;
      }
      if (um2::abs(cos_ia_theta) < eps) {
        cos_ia_theta = 0;
      }
      Float const rx = um2::abs(pitch / (2 * cos_ia_theta));
      Float const ry = um2::abs(pitch / (2 * sin_ia_theta));
      Float const rb = um2::min(rx, ry);
      vertices[num_prev_points + ia] = {rb * cos_ia_theta, rb * sin_ia_theta};
    }

    //------------------------------------------------------------------------
    // Get the faces that make up the mesh
    //------------------------------------------------------------------------
    Int const num_faces = num_azimuthal * (total_rings + 1);
    Vector<Vec<4, Int>> faces(num_faces);
    // Establish a few aliases
    Int const na = num_azimuthal;
    Int const nr = total_rings;
    Int const ntric = 1 + na / 2; // Number of triangular points + center point
    // Triangular quads
    for (Int ia = 0; ia < na / 2; ++ia) {
      Int const p0 = 0;                  // Center point
      Int const p1 = ntric + ia * 2;     // Bottom right point on ring
      Int const p2 = ntric + ia * 2 + 1; // Top right point on ring
      Int const p3 = 1 + ia;             // The triangular point
      Int p4 = ntric + ia * 2 + 2;       // Top left point on ring
      // If we're at the end of the ring, wrap around
      if (p4 == ntric + na) {
        p4 = ntric;
      }
      faces[2 * ia] = {p0, p1, p2, p3};
      faces[2 * ia + 1] = {p0, p3, p2, p4};
    }
    // Non-boundary and boundary quads
    for (Int ir = 1; ir < nr + 1; ++ir) {
      for (Int ia = 0; ia < na; ++ia) {
        Int const p0 = ntric + (ir - 1) * na + ia; // Bottom left point
        Int const p1 = ntric + (ir)*na + ia;       // Bottom right point
        Int p2 = ntric + (ir)*na + ia + 1;         // Top right point
        Int p3 = ntric + (ir - 1) * na + ia + 1;   // Top left point
        // If we're at the end of the ring, wrap around
        if (ia + 1 == na) {
          p2 -= na;
          p3 -= na;
        }
        faces[ir * na + ia] = {p0, p1, p2, p3};
      }
    }
    // Shift such that the lower left corner is at the origin
    Float const half_pitch = pitch / 2;
    for (Int i = 0; i < num_points; ++i) {
      vertices[i] += half_pitch;
      // Fix close to zero values
      if (um2::abs(vertices[i][0]) < eps) {
        vertices[i][0] = 0;
      }
      if (um2::abs(vertices[i][1]) < eps) {
        vertices[i][1] = 0;
      }
    }
    this->_quads.emplace_back(vertices, faces);
    LOG_DEBUG("Finished creating mesh");
    return mesh_id;
  }
  if (mesh_order == 2) {
    // Get the equivalent radius of each ring if it were a quadratic quadrilateral
    Float const theta = 2 * pi<Float> / num_azimuthal_t;
    Float const gamma = theta / 2;
    Float const sin_gamma = um2::sin(gamma);
    Float const cos_gamma = um2::cos(gamma);
    Float const sincos_gamma = sin_gamma * cos_gamma;
    Vector<Float> eq_radii(total_rings);
    // The innermost radius is a special case, and is essentially a triangle.
    // Each quadratic shape is made up of the linear shape plus quadratic edges
    // A_t = l² * sin(θ) / 2 = l² * sin(θ/2) * cos(θ/2)
    // A_q = (l² - l²₀) * sin(θ) / 2 = (l² - l²₀) * sin(θ/2) * cos(θ/2)
    // A_edge = (4 / 3) the area of the triangle formed by the vertices of the edge.
    //        = (4 / 3) * 2l sin(θ/2) * (L - l cos(θ/2)) / 2
    //        = (4 / 3) * l sin(θ/2) * (L - l cos(θ/2))
    //
    // For N + 1 rings
    // A_0 = pi R_0² = Na ( A_t + A_e0)
    // A_i = pi (R_i² - R_{i-1}²) = Na ( A_q + A_ei - A_ei-1)
    // A_N = P² - pi R_N² = P² - sum_i=0^N A_i
    // Constraining L_N to be the value which minimizes the 2-norm of the integral of
    // the quadratic segment minus the circle arc is the correct thing to do, but holy
    // cow the integral is a mess.
    // Therefore we settle for constraining l_i = r_i
    Float const tri_area = ring_radii[0] * ring_radii[0] * sincos_gamma;
    Float const ring_area = ring_areas[0] / num_azimuthal_t;
    Float const l0 = ring_radii[0];
    Float constexpr three_fourths = static_cast<Float>(3) / static_cast<Float>(4);
    eq_radii[0] =
        three_fourths * (ring_area - tri_area) / (l0 * sin_gamma) + l0 * cos_gamma;
    for (Int i = 1; i < total_rings; ++i) {
      Float const l_im1 = ring_radii[i - 1];
      Float const ll_im1 = eq_radii[i - 1];
      Float const a_edge_im1 =
          l_im1 * sin_gamma * (ll_im1 - l_im1 * cos_gamma) / three_fourths;
      Float const l = ring_radii[i];
      Float const a_quad = (l * l - l_im1 * l_im1) * sincos_gamma;
      Float const a_ring = ring_areas[i] / num_azimuthal_t;
      eq_radii[i] = three_fourths * (a_ring - a_quad + a_edge_im1) / (l * sin_gamma) +
                    l * cos_gamma;
    }
    // Log the equivalent radii in debug mode
    for (Int i = 0; i < total_rings; ++i) {
      LOG_DEBUG("Ring ", i, " equivalent radius: ", eq_radii[i]);
    }
    // If any of the equivalent radii are larger than half the pitch, error
    if (std::any_of(eq_radii.begin(), eq_radii.end(),
                    [pitch](Float r) { return r > pitch / 2; })) {
      logger::error("The equivalent radius of a ring is larger than half the pitch.");
      return -1;
    }

    //-------------------------------------------------------------------------
    // Get the points that make up the mesh
    //-------------------------------------------------------------------------
    // The number of points is:
    //   Center point
    //   2 * num_azimuthal for the triangular points inside the first ring
    //   2 * num_azimuthal for the triangular points on the first ring
    //   3 * num_azimuthal * total_rings
    //
    // Some aliases to make the code more readable
    Int const na = num_azimuthal;
    Int const nr = total_rings;
    Int const num_points = 1 + 4 * na + 3 * na * nr;
    Vector<Point2F> vertices(num_points);
    // Center point
    vertices[0] = {0, 0};
    // Triangular points
    Float const rt = ring_radii[0] / 2;
    for (Int ia = 0; ia < na; ++ia) {
      Float const sin_ia_theta = um2::sin(static_cast<Float>(ia) * theta);
      Float const cos_ia_theta = um2::cos(static_cast<Float>(ia) * theta);
      // if ia is 0 or even, just do the 1 center point, otherwise we need 3 points
      // at (1/4, 2/4, 3/4) of the radius
      if (ia % 2 == 0) {
        vertices[1 + 2 * ia] = {rt * cos_ia_theta, rt * sin_ia_theta};
      } else {
        vertices[2 * ia] = {rt * cos_ia_theta / 2, rt * sin_ia_theta / 2};
        vertices[2 * ia + 1] = {rt * cos_ia_theta, rt * sin_ia_theta};
        vertices[2 * ia + 2] = {3 * rt * cos_ia_theta / 2, 3 * rt * sin_ia_theta / 2};
      }
    }
    // Points on the first ring
    Int num_prev_points = 1 + 2 * na;
    for (Int ia = 0; ia < 2 * na; ++ia) {
      Float sin_ia_gamma = um2::sin(static_cast<Float>(ia) * gamma);
      Float cos_ia_gamma = um2::cos(static_cast<Float>(ia) * gamma);
      if (um2::abs(sin_ia_gamma) < eps) {
        sin_ia_gamma = 0;
      }
      if (um2::abs(cos_ia_gamma) < eps) {
        cos_ia_gamma = 0;
      }
      // if ia is 0 or even, we want the point at ring_radii[ir], otherwise we want
      // the point at eq_radii[ir]
      if (ia % 2 == 0) {
        vertices[num_prev_points + ia] = {ring_radii[0] * cos_ia_gamma,
                                          ring_radii[0] * sin_ia_gamma};
      } else {
        vertices[num_prev_points + ia] = {eq_radii[0] * cos_ia_gamma,
                                          eq_radii[0] * sin_ia_gamma};
      }
    }
    // Points on and between the rings
    for (Int ir = 1; ir < total_rings; ++ir) {
      num_prev_points = 1 + 4 * na + 3 * na * (ir - 1);
      // Between the rings
      for (Int ia = 0; ia < num_azimuthal; ++ia) {
        Float sin_ia_theta = um2::sin(static_cast<Float>(ia) * theta);
        Float cos_ia_theta = um2::cos(static_cast<Float>(ia) * theta);
        if (um2::abs(sin_ia_theta) < eps) {
          sin_ia_theta = 0;
        }
        if (um2::abs(cos_ia_theta) < eps) {
          cos_ia_theta = 0;
        }
        Float const r = (ring_radii[ir] + ring_radii[ir - 1]) / 2;
        vertices[num_prev_points + ia] = {r * cos_ia_theta, r * sin_ia_theta};
      }
      num_prev_points += num_azimuthal;
      for (Int ia = 0; ia < 2 * num_azimuthal; ++ia) {
        Float sin_ia_gamma = um2::sin(static_cast<Float>(ia) * gamma);
        Float cos_ia_gamma = um2::cos(static_cast<Float>(ia) * gamma);
        if (um2::abs(sin_ia_gamma) < eps) {
          sin_ia_gamma = 0;
        }
        if (um2::abs(cos_ia_gamma) < eps) {
          cos_ia_gamma = 0;
        }
        // if ia is 0 or even, we want the point at ring_radii[ir], otherwise we
        // want the point at eq_radii[ir]
        if (ia % 2 == 0) {
          vertices[num_prev_points + ia] = {ring_radii[ir] * cos_ia_gamma,
                                            ring_radii[ir] * sin_ia_gamma};
        } else {
          vertices[num_prev_points + ia] = {eq_radii[ir] * cos_ia_gamma,
                                            eq_radii[ir] * sin_ia_gamma};
        }
      }
    }
    // Quadratic points before the boundary
    num_prev_points = 1 + 4 * na + 3 * na * (total_rings - 1);
    for (Int ia = 0; ia < num_azimuthal; ++ia) {
      Float sin_ia_theta = um2::sin(static_cast<Float>(ia) * theta);
      Float cos_ia_theta = um2::cos(static_cast<Float>(ia) * theta);
      if (um2::abs(sin_ia_theta) < eps) {
        sin_ia_theta = 0;
      }
      if (um2::abs(cos_ia_theta) < eps) {
        cos_ia_theta = 0;
      }
      // pitch and last ring radius
      Float const rx = um2::abs(pitch / (2 * cos_ia_theta));
      Float const ry = um2::abs(pitch / (2 * sin_ia_theta));
      Float const rb = um2::min(rx, ry);
      Float const r = (rb + ring_radii[total_rings - 1]) / 2;
      vertices[num_prev_points + ia] = {r * cos_ia_theta, r * sin_ia_theta};
    }
    // Points on the boundary of the pin (pitch / 2)
    num_prev_points += num_azimuthal;
    for (Int ia = 0; ia < 2 * num_azimuthal; ++ia) {
      Float sin_ia_gamma = um2::sin(gamma * static_cast<Float>(ia));
      Float cos_ia_gamma = um2::cos(gamma * static_cast<Float>(ia));
      if (um2::abs(sin_ia_gamma) < eps) {
        sin_ia_gamma = 0;
      }
      if (um2::abs(cos_ia_gamma) < eps) {
        cos_ia_gamma = 0;
      }
      Float const rx = um2::abs(pitch / (2 * cos_ia_gamma));
      Float const ry = um2::abs(pitch / (2 * sin_ia_gamma));
      Float const rb = um2::min(rx, ry);
      vertices[num_prev_points + ia] = {rb * cos_ia_gamma, rb * sin_ia_gamma};
    }
    for (Int i = 0; i < num_points; ++i) {
      LOG_DEBUG("Point ", i, ": ", vertices[i][0], ", ", vertices[i][1]);
    }

    //-------------------------------------------------------------------------
    // Get the faces that make up the mesh
    //-------------------------------------------------------------------------
    Int const num_faces = na * (nr + 1);
    Vector<Vec<8, Int>> faces(num_faces);
    // Triangular quads
    for (Int ia = 0; ia < na / 2; ++ia) {
      Int const p0 = 0;                   // Center point
      Int const p1 = 1 + 2 * na + 4 * ia; // Bottom right point on ring
      Int const p2 = p1 + 2;              // Top right point on ring
      Int const p3 = 3 + 4 * ia;          // The triangular point
      Int p4 = p2 + 2;                    // Top left point on ring
      Int const p5 = 1 + 4 * ia;          // Bottom quadratic point
      Int const p6 = p1 + 1;              // Right quadratic point
      Int const p7 = p3 + 1;              // Top tri quadratic point
      Int const p8 = p3 - 1;              // Bottom tri quadratic point
      Int const p9 = p2 + 1;              // Top right quadratic point
      Int p10 = p7 + 1;                   // Top left quadratic point
      // If we're at the end of the ring, wrap around
      if (p10 == 1 + 2 * na) {
        p4 -= 2 * na;
        p10 -= 2 * na;
      }
      faces[2 * ia] = {p0, p1, p2, p3, p5, p6, p7, p8};
      faces[2 * ia + 1] = {p0, p3, p2, p4, p8, p7, p9, p10};
    }
    // All other faces
    for (Int ir = 1; ir < nr + 1; ++ir) {
      Int const np = 1 + 2 * na + 3 * na * (ir - 1);
      for (Int ia = 0; ia < na; ++ia) {
        Int const p0 = np + 2 * ia;
        Int const p1 = p0 + 3 * na;
        Int p2 = p1 + 2;
        Int p3 = p0 + 2;
        Int const p4 = np + 2 * na + ia;
        Int const p5 = p1 + 1;
        Int p6 = p4 + 1;
        Int const p7 = p0 + 1;
        // If we're at the end of the ring, wrap around
        if (ia + 1 == na) {
          p2 -= 2 * na;
          p3 -= 2 * na;
          p6 -= na;
        }
        faces[ir * na + ia] = {p0, p1, p2, p3, p4, p5, p6, p7};
      }
    }
    // Print the faces
    for (Int i = 0; i < num_faces; ++i) {
      LOG_DEBUG("Face ", i, ":", faces[i][0], faces[i][1], faces[i][2], faces[i][3],
                faces[i][4], faces[i][5], faces[i][6], faces[i][7]);
    }

    // Shift such that the lower left corner is at the origin
    Float const half_pitch = pitch / 2;
    for (Int i = 0; i < num_points; ++i) {
      vertices[i] += half_pitch;
      // Fix close to zero values
      if (um2::abs(vertices[i][0]) < eps) {
        vertices[i][0] = 0;
      }
      if (um2::abs(vertices[i][1]) < eps) {
        vertices[i][1] = 0;
      }
    }
    this->_quad8s.emplace_back(vertices, faces);
    LOG_DEBUG("Finished creating mesh");
    return mesh_id;
  }
  logger::error(
      "Only linear and quadratic meshes are supported for a cylindrical pin mesh");
  return -1;
}

//=============================================================================
// addCylindricalPinCell
//=============================================================================

auto
Model::addCylindricalPinCell(Float const pitch, Vector<Float> const & radii,
                             Vector<Material> const & materials,
                             Vector<Int> const & num_rings, Int const num_azimuthal,
                             Int const mesh_order) -> Int
{
  // Make the mesh
  Int const mesh_id =
      addCylindricalPinMesh(pitch, radii, num_rings, num_azimuthal, mesh_order);

  // We need 1 more material than the number of rings
  if (materials.size() != num_rings.size() + 1) {
    logger::error("The number of materials must be one more than the number of rings");
    return -1;
  }

  // Get the index of each of the materials
  Vector<Int> mat_idx(materials.size());
  for (Int imat = 0; imat < materials.size(); ++imat) {
    bool found = false;
    for (Int i = 0; i < _materials.size(); ++i) {
      if (_materials[i].getName() == materials[imat].getName()) {
        mat_idx[imat] = i;
        found = true;
        break;
      }
    }
    if (!found) {
      logger::error("Material ", materials[imat].getName(), " not found in model");
      return -1;
    }
  }

  Vec2F const xy_extents(pitch, pitch);
  MeshType const mesh_type = mesh_order == 1 ? MeshType::Quad : MeshType::QuadraticQuad;

  Int const total_rings = std::reduce(num_rings.cbegin(), num_rings.cend(), 0);
  Vector<MatID> material_ids((total_rings + 1) * num_azimuthal);

  // For each material, get the number of faces (num_azimuthal * num_rings)
  Int ctr = 0;
  for (Int imat = 0; imat < materials.size() - 1; ++imat) {
    for (Int ir = 0; ir < num_rings[imat]; ++ir) {
      for (Int ia = 0; ia < num_azimuthal; ++ia, ++ctr) {
        material_ids[ctr] = static_cast<MatID>(mat_idx[imat]);
      }
    }
  }
  // Last faces outside the last ring
  for (Int ia = 0; ia < num_azimuthal; ++ia, ++ctr) {
    material_ids[ctr] = static_cast<MatID>(mat_idx.back());
  }

  return addCoarseCell(xy_extents, mesh_type, mesh_id, material_ids);
}

//=============================================================================
// addRectangularPinMesh
//=============================================================================

auto
Model::addRectangularPinMesh(Vec2F const xy_extents, Int const nx_faces,
                             Int const ny_faces) -> Int
{
  Int const mesh_id = _quads.size();
  LOG_DEBUG("Adding rectangular pin mesh ", mesh_id);
  _quads.emplace_back();
  auto & mesh = _quads.back();

  if (xy_extents[0] <= 0 || xy_extents[1] <= 0) {
    logger::error("Pin dimensions must be positive");
    return -1;
  }
  if (nx_faces <= 0 || ny_faces <= 0) {
    logger::error("Number of faces must be positive");
    return -1;
  }

  // Make the vertices
  auto & vertices = mesh.vertices();
  vertices.resize((nx_faces + 1) * (ny_faces + 1));
  Float const dx = xy_extents[0] / static_cast<Float>(nx_faces);
  Float const dy = xy_extents[1] / static_cast<Float>(ny_faces);
  for (Int j = 0; j <= ny_faces; ++j) {
    Float const y = static_cast<Float>(j) * dy;
    for (Int i = 0; i <= nx_faces; ++i) {
      Float const x = static_cast<Float>(i) * dx;
      vertices[j * (nx_faces + 1) + i] = {x, y};
    }
  }
  // Make the faces
  auto & face_conn = mesh.faceVertexConn();
  face_conn.resize(nx_faces * ny_faces);
  // Left to right, bottom to top
  for (Int j = 0; j < ny_faces; ++j) {
    for (Int i = 0; i < nx_faces; ++i) {
      // v3 -- v2
      // |     |
      // v0 -- v1
      Int const v0 = j * (nx_faces + 1) + i;
      Int const v1 = v0 + 1;
      Int const v2 = v1 + nx_faces + 1;
      Int const v3 = v2 - 1;
      face_conn[j * nx_faces + i][0] = v0;
      face_conn[j * nx_faces + i][1] = v1;
      face_conn[j * nx_faces + i][2] = v2;
      face_conn[j * nx_faces + i][3] = v3;
    }
  }
  mesh.validate();
  return mesh_id;
}

//=============================================================================
// addMesh
//=============================================================================

auto
Model::addTriMesh(TriFVM const & mesh) -> Int
{
  Int const mesh_id = _tris.size();
  LOG_DEBUG("Adding triangular mesh ", mesh_id);
  ASSERT(mesh.boundingBox().minima().norm() < epsDistance<Float>());
  _tris.emplace_back(mesh);
  return mesh_id;
}

auto
Model::addQuadMesh(QuadFVM const & mesh) -> Int
{
  Int const mesh_id = _quads.size();
  LOG_DEBUG("Adding quadrilateral mesh ", mesh_id);
  ASSERT(mesh.boundingBox().minima().norm() < epsDistance<Float>());
  _quads.emplace_back(mesh);
  return mesh_id;
}

auto
Model::addTri6Mesh(Tri6FVM const & mesh) -> Int
{
  Int const mesh_id = _tri6s.size();
  LOG_DEBUG("Adding quadratic triangular mesh ", mesh_id);
  ASSERT(mesh.boundingBox().minima().norm() < epsDistance<Float>());
  _tri6s.emplace_back(mesh);
  return mesh_id;
}

auto
Model::addQuad8Mesh(Quad8FVM const & mesh) -> Int
{
  Int const mesh_id = _quad8s.size();
  LOG_DEBUG("Adding quadratic quadrilateral mesh ", mesh_id);
  ASSERT(mesh.boundingBox().minima().norm() < epsDistance<Float>());
  _quad8s.emplace_back(mesh);
  return mesh_id;
}

//=============================================================================
// addCoarseCell
//=============================================================================

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::addCoarseCell(Vec2F const xy_extents, MeshType const mesh_type, Int const mesh_id,
                     Vector<MatID> const & material_ids) -> Int
{
  Int const cc_id = _coarse_cells.size();
  LOG_DEBUG("Adding coarse cell ", cc_id);
  // Ensure extents are positive
  if (xy_extents[0] <= 0 || xy_extents[1] <= 0) {
    logger::error("Coarse cell dimensions must be positive");
    return -1;
  }
  // Ensure that the mesh exists and the material IDs are valid
  if (mesh_id != -1) {
    switch (mesh_type) {
    case MeshType::Tri:
      if (0 > mesh_id || mesh_id >= _tris.size()) {
        logger::error("Tri mesh ", mesh_id, " does not exist");
        return -1;
      }
      if (_tris[mesh_id].numFaces() != material_ids.size()) {
        logger::error("Mismatch between number of faces and provided materials");
        return -1;
      }
      break;
    case MeshType::Quad:
      if (0 > mesh_id || mesh_id >= _quads.size()) {
        logger::error("Quad mesh ", mesh_id, " does not exist");
        return -1;
      }
      if (_quads[mesh_id].numFaces() != material_ids.size()) {
        logger::error("Mismatch between number of faces and provided materials");
        return -1;
      }
      break;
    case MeshType::QuadraticTri:
      if (0 > mesh_id || mesh_id >= _tri6s.size()) {
        logger::error("Quadratic tri mesh ", mesh_id, " does not exist");
        return -1;
      }
      if (_tri6s[mesh_id].numFaces() != material_ids.size()) {
        logger::error("Mismatch between number of faces and provided materials");
        return -1;
      }
      break;
    case MeshType::QuadraticQuad:
      if (0 > mesh_id || mesh_id >= _quad8s.size()) {
        logger::error("Quadratic quad mesh ", mesh_id, " does not exist");
        return -1;
      }
      if (_quad8s[mesh_id].numFaces() != material_ids.size()) {
        logger::error("Mismatch between number of faces and provided materials");
        return -1;
      }
      break;
    default:
      logger::error("Invalid mesh type");
      return -1;
    }
    // Check that the material IDs are valid
    for (auto const & mat_id : material_ids) {
      if (mat_id < 0 || mat_id >= _materials.size()) {
        logger::error("Material ", mat_id, " does not exist");
        return -1;
      }
    }
  }
  // Create the coarse cell
  CoarseCell cc;
  cc.xy_extents = xy_extents;
  cc.mesh_type = mesh_type;
  cc.mesh_id = mesh_id;
  cc.material_ids = material_ids;
  _coarse_cells.emplace_back(um2::move(cc));
  return cc_id;
}

//=============================================================================
// addRTM
//=============================================================================

auto
Model::addRTM(Vector<Vector<Int>> const & cc_ids) -> Int
{
  Int const rtm_id = _rtms.size();
  LOG_DEBUG("Adding ray tracing module ", rtm_id);
  Vector<Int> unique_cc_ids;
  Vector<Vec2F> xy_extents;
  // Ensure that all coarse cells exist
  Int const num_cc = _coarse_cells.size();
  for (auto const & cc_ids_row : cc_ids) {
    for (auto const & id : cc_ids_row) {
      if (id < 0 || id >= num_cc) {
        logger::error("Coarse cell ", id, " does not exist");
        return -1;
      }
      auto * const it = std::find(unique_cc_ids.begin(), unique_cc_ids.end(), id);
      if (it == unique_cc_ids.end()) {
        unique_cc_ids.emplace_back(id);
        xy_extents.emplace_back(_coarse_cells[id].xy_extents);
      }
    }
  }
  // For a max pin ID N, the RectilinearGrid constructor needs all dxdy from 0 to N.
  // To get around this requirement, we will renumber the coarse cells to be 0, 1, 2,
  // 3, ..., and then use the renumbered IDs to create the RectilinearGrid.
  Vector<Vector<Int>> cc_ids_renumbered(cc_ids.size());
  for (Int i = 0; i < cc_ids.size(); ++i) {
    cc_ids_renumbered[i].resize(cc_ids[i].size());
    for (Int j = 0; j < cc_ids[i].size(); ++j) {
      auto * const it =
          std::find(unique_cc_ids.begin(), unique_cc_ids.end(), cc_ids[i][j]);
      ASSERT(it != unique_cc_ids.cend());
      cc_ids_renumbered[i][j] = static_cast<Int>(it - unique_cc_ids.begin());
    }
  }
  // Create the rectilinear grid
  RectilinearGrid2F const grid(xy_extents, cc_ids_renumbered);
  // Ensure the grid has the same extents as all other RTMs
  if (!_rtms.empty()) {
    if (!_rtms[0].grid().extents().isApprox(grid.extents())) {
      logger::error("All RTMs must have the same extents");
      return -1;
    }
  }
  // Flatten the coarse cell IDs (rows are reversed)
  Vector<Int> cc_ids_flat;
  flattenLattice(cc_ids, cc_ids_flat);
  RTM rtm(grid, cc_ids_flat);
  _rtms.push_back(um2::move(rtm));
  return rtm_id;
}

//=============================================================================
// addLattice
//=============================================================================

auto
Model::addLattice(Vector<Vector<Int>> const & rtm_ids) -> Int
{
  Int const lat_id = _lattices.size();
  LOG_DEBUG("Adding lattice ", lat_id);
  // Ensure that all RTMs exist
  Int const num_rtm = _rtms.size();
  for (auto const & rtm_ids_row : rtm_ids) {
    for (auto const & id : rtm_ids_row) {
      if (id < 0 || id >= num_rtm) {
        logger::error("RTM ", id, " does not exist");
        return -1;
      }
    }
  }
  // Create the lattice
  // Ensure each row has the same number of columns
  Point2F const minima(0, 0);
  Vec2F const spacing = _rtms[0].grid().extents();
  Int const num_rows = rtm_ids.size();
  Int const num_cols = rtm_ids[0].size();
  for (Int i = 1; i < num_rows; ++i) {
    if (rtm_ids[i].size() != num_cols) {
      logger::error("Each row must have the same number of columns");
      return -1;
    }
  }
  Vec2I const num_cells(num_cols, num_rows);
  RegularGrid2F const grid(minima, spacing, num_cells);
  // Flatten the RTM IDs (rows are reversed)
  Vector<Int> rtm_ids_flat;
  flattenLattice(rtm_ids, rtm_ids_flat);
  Lattice lat(grid, rtm_ids_flat);
  _lattices.push_back(um2::move(lat));
  return lat_id;
}

//=============================================================================
// addAssembly
//=============================================================================

auto
Model::addAssembly(Vector<Int> const & lat_ids, Vector<Float> const & z) -> Int
{
  Int const asy_id = _assemblies.size();
  LOG_DEBUG("Adding assembly ", asy_id);
  // Ensure that all lattices exist
  Int const num_lat = _lattices.size();
  for (auto const & id : lat_ids) {
    if (id < 0 || id >= num_lat) {
      logger::error("Lattice ", id, " does not exist");
      return -1;
    }
  }
  // Ensure the number of lattices is 1 less than the number of z-planes
  if (lat_ids.size() + 1 != z.size()) {
    logger::error("The number of lattices must be 1 less than the number of z-planes");
    return -1;
  }
  // Ensure all z-planes are in ascending order
  if (!um2::is_sorted(z.begin(), z.end())) {
    logger::error("The z-planes must be in ascending order");
    return -1;
  }
  // Unless this is the 2D special case, ensure all z-coordinates are positive
  auto const ahalf = castIfNot<Float>(0.5);
  // Check if z = {-ahalf, ahalf} for the 2D special case
  if (um2::abs(z.front() + ahalf) > epsDistance<Float>() ||
      um2::abs(z.back() - ahalf) > epsDistance<Float>()) {
    if (um2::abs(z.front()) > epsDistance<Float>()) {
      logger::error("The first z-plane must be at 0");
      return -1;
    }
  }
  // Ensure this assembly is the same height as all other assemblies
  if (!_assemblies.empty()) {
    auto const assem_top = _assemblies[0].grid().maxima(0);
    auto const assem_bot = _assemblies[0].grid().minima(0);
    if (um2::abs(z.back() - assem_top) > epsDistance<Float>() ||
        um2::abs(z.front() - assem_bot) > epsDistance<Float>()) {
      logger::error("All assemblies must have the same height");
      return -1;
    }
  }
  // Ensure the lattices all have the same dimensions. Since they are composed of RTMs,
  // it is sufficient to check numCells(0) and numCells(1).
  Int const num_xcells = _lattices[lat_ids[0]].grid().numCells(0);
  Int const num_ycells = _lattices[lat_ids[0]].grid().numCells(1);
  for (auto const lat_it : lat_ids) {
    if (_lattices[lat_it].grid().numCells(0) != num_xcells ||
        _lattices[lat_it].grid().numCells(1) != num_ycells) {
      logger::error("All lattices in the assembly must have the same xy-dimensions");
      return -1;
    }
  }

  RectilinearGrid1F grid;
  grid.divs(0).resize(z.size());
  um2::copy(z.cbegin(), z.cend(), grid.divs(0).begin());
  Assembly asy(grid, lat_ids);
  _assemblies.emplace_back(um2::move(asy));
  return asy_id;
}

//=============================================================================
// addCore
//=============================================================================

auto
Model::addCore(Vector<Vector<Int>> const & asy_ids) -> Int
{
  LOG_DEBUG("Adding core");
  // Ensure it is not already made
  if (!_core.children().empty()) {
    logger::error("The core has already been made");
    return -1;
  }

  // Ensure that all assemblies exist
  Int const num_asy = _assemblies.size();
  for (auto const & asy_ids_row : asy_ids) {
    for (auto const & id : asy_ids_row) {
      if (id < 0 || id >= num_asy) {
        logger::error("Assembly ", id, " does not exist");
        return -1;
      }
    }
  }
  Vector<Vec2F> xy_extents(num_asy);
  for (Int i = 0; i < num_asy; ++i) {
    auto const lat_id = _assemblies[i].getChild(0);
    xy_extents[i] = _lattices[lat_id].grid().extents();
  }
  // Create the rectilinear grid
  RectilinearGrid2F const grid(xy_extents, asy_ids);
  // Flatten the assembly IDs (rows are reversed)
  Vector<Int> asy_ids_flat;
  flattenLattice(asy_ids, asy_ids_flat);
  Core core(grid, asy_ids_flat);
  _core = um2::move(core);
  return 0;
}

//=============================================================================
// addCoarseGrid
//=============================================================================

void
Model::addCoarseGrid(Vec2F const xy_extents, Vec2I const xy_num_cells)
{
  LOG_INFO("Adding an ", xy_num_cells[0], " x ", xy_num_cells[1], " coarse grid");
  if (xy_extents[0] <= 0 || xy_extents[1] <= 0) {
    logger::error("Grid dimensions must be positive");
    return;
  }

  if (xy_num_cells[0] <= 0 || xy_num_cells[1] <= 0) {
    logger::error("Number of cells must be positive");
    return;
  }

  // Ensure that the model is empty
  ASSERT(_coarse_cells.empty());
  ASSERT(_rtms.empty());
  ASSERT(_lattices.empty());
  ASSERT(_assemblies.empty());

  Float const dx = xy_extents[0] / static_cast<Float>(xy_num_cells[0]);
  Float const dy = xy_extents[1] / static_cast<Float>(xy_num_cells[1]);
  Vec2F const dxdy(dx, dy);
  Int const nx = xy_num_cells[0];
  Int const ny = xy_num_cells[1];

  // Add a coarse cell for each cell in the grid
  for (Int iy = 0; iy < ny; ++iy) {
    for (Int ix = 0; ix < nx; ++ix) {
      addCoarseCell(dxdy);
    }
  }

  // Add an RTM for each coarse cell
  Vector<Vector<Int>> ids = {{0}};
  for (Int i = 0; i < nx * ny; ++i) {
    ids[0][0] = i;
    addRTM(ids);
  }

  // Add a single lattice with all the RTMs
  // The ids in the rows are reversed, so we start from max row
  ids.resize(ny);
  for (Int iy = ny - 1; iy >= 0; --iy) {
    ids[iy].resize(nx);
    for (Int ix = 0; ix < nx; ++ix) {
      ids[iy][ix] = (ny - 1 - iy) * nx + ix;
    }
  }
  addLattice(ids);

  // Add a single assembly with the lattice
  addAssembly({0});

  // Add the core with the assembly
  addCore({{0}});
}

//=============================================================================
// importCoarseCellMeshes
//=============================================================================

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::importCoarseCellMeshes(String const & filename)
{
  LOG_INFO("Importing coarse cells from ", filename);
  ASSERT(!_materials.empty());

  PolytopeSoup const soup(filename);

  String cc_name("Coarse_Cell_00000");

  // For each coarse cell
  Int const num_coarse_cells = numCoarseCells();
  for (Int icc = 0; icc < num_coarse_cells; ++icc) {

    // Get the mesh for the coarse cell
    PolytopeSoup cc_mesh;
    soup.getSubset(cc_name, cc_mesh);
    incrementASCIINumber(cc_name);

    // Get the mesh type and material IDs
    MeshType const mesh_type = getMeshType(cc_mesh.getElemTypes());
    ASSERT(mesh_type != MeshType::Invalid);
    ASSERT(mesh_type != MeshType::TriQuad);
    ASSERT(mesh_type != MeshType::QuadraticTriQuad);
    CoarseCell & cc = _coarse_cells[icc];
    cc.mesh_type = mesh_type;
    cc.material_ids.resize(cc_mesh.numElements());
    um2::fill(cc.material_ids.begin(), cc.material_ids.end(), static_cast<MatID>(-1));
    // For each elset in the cc_mesh, test if it is a material.
    // If so, set all the elements in the elset to the corresponding material ID.
    for (auto const & elset_name : cc_mesh.elsetNames()) {
      if (elset_name.starts_with("Material_")) {
        String const mat_name = elset_name.substr(9);
        // Get the material ID (index into the materials vector)
        bool mat_found = false;
        for (Int imat = 0; imat < _materials.size(); ++imat) {
          if (_materials[imat].getName() == mat_name) {
            mat_found = true;
            Vector<Int> ids;
            Vector<Float> data;
            cc_mesh.getElset(elset_name, ids, data);
            ASSERT(!ids.empty());
            for (Int const & id : ids) {
              ASSERT(id >= 0);
              ASSERT(id < cc_mesh.numElements());
              cc.material_ids[id] = static_cast<MatID>(imat);
            }
            break;
          }
        }
        if (!mat_found) {
          logger::error("Material ", elset_name, " not found");
          return;
        }
      }
    }
    // Check that no material IDs are -1
    for (auto const & id : cc.material_ids) {
      if (id == -1) {
        logger::error("mesh does not have a material ID for each face");
        return;
      }
    }
    // Create the FaceVertexMesh and shift it from global coordinates to local
    // coordinates, with the bottom left corner of the AABB at the origin
    AxisAlignedBox2F bb = AxisAlignedBox2F::empty();
    Point2F * vertices = nullptr;
    Int const num_verts = cc_mesh.numVertices();
    switch (mesh_type) {
    case MeshType::Tri:
      cc.mesh_id = _tris.size();
      _tris.emplace_back(cc_mesh);
      bb = _tris.back().boundingBox();
      vertices = _tris.back().vertices().data();
      break;
    case MeshType::Quad:
      cc.mesh_id = _quads.size();
      _quads.emplace_back(cc_mesh);
      bb = _quads.back().boundingBox();
      vertices = _quads.back().vertices().data();
      break;
    case MeshType::QuadraticTri:
      cc.mesh_id = _tri6s.size();
      _tri6s.emplace_back(cc_mesh);
      bb = _tri6s.back().boundingBox();
      vertices = _tri6s.back().vertices().data();
      break;
    case MeshType::QuadraticQuad:
      cc.mesh_id = _quad8s.size();
      _quad8s.emplace_back(cc_mesh);
      bb = _quad8s.back().boundingBox();
      vertices = _quad8s.back().vertices().data();
      break;
    default:
      logger::error("Mesh type not supported");
      return;
    }

    // Shift the points so that the min point is at the origin.
    Point2F const min_point = bb.minima();
    for (Int ip = 0; ip < num_verts; ++ip) {
      vertices[ip] -= min_point;
    }
#if UM2_ENABLE_ASSERTS
    Point2F const dxdy = bb.maxima() - bb.minima();
    ASSERT(dxdy.isApprox(cc.xy_extents));
#endif
  }
} // importCoarseCellMeshes

//=============================================================================
// operator PolytopeSoup
//=============================================================================

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::operator PolytopeSoup() const noexcept
{
  LOG_DEBUG("Converting MPACT model to PolytopeSoup");

  PolytopeSoup core_soup;

  if (_core.children().empty()) {
    logger::error("Core has no children");
    return core_soup;
  }

  // Store the one-group cross sections for each material
  Vector<Float> one_group_xs(_materials.size());
  for (Int imat = 0; imat < _materials.size(); ++imat) {
    ASSERT(_materials[imat].xsec().isMacro());
    auto const xs = _materials[imat].xsec().collapseTo1GroupAvg();
    one_group_xs[imat] = xs.t(0);
  }

  // Allocate counters for each assembly, lattice, etc.
  Vector<Int> asy_found(_assemblies.size(), -1);
  Vector<Int> lat_found(_lattices.size(), -1);
  Vector<Int> rtm_found(_rtms.size(), -1);
  Vector<Int> cc_found(_coarse_cells.size(), -1);

  // For each assembly
  Int const nyasy = _core.grid().numCells(1);
  Int const nxasy = _core.grid().numCells(0);
  for (Int iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Int ixasy = 0; ixasy < nxasy; ++ixasy) {

      PolytopeSoup assembly_soup;

      // Get the assembly ID
      auto const asy_id = _core.getChild(ixasy, iyasy);
      ASSERT(asy_id >= 0);
      ASSERT(asy_id < _assemblies.size());

      // Increment the number of times we have seen this assembly
      Int const asy_id_ctr = ++asy_found[asy_id];

      // Get the assembly name
      String const asy_name =
          "Assembly_" + getASCIINumber(asy_id) + "_" + getASCIINumber(asy_id_ctr);
      LOG_DEBUG("Assembly name: ", asy_name);

      // Get the assembly offset (lower left corner)
      Point2F const asy_ll = _core.grid().getBox(ixasy, iyasy).minima();

      // Get the assembly
      auto const & assembly = _assemblies[asy_id];
      if (assembly.children().empty()) {
        logger::error("Assembly has no children");
        return core_soup;
      }

      // For each lattice
      Int const nzlat = assembly.grid().numCells(0);
      for (Int izlat = 0; izlat < nzlat; ++izlat) {

        PolytopeSoup lattice_soup;

        // Get the lattice ID
        auto const lat_id = assembly.getChild(izlat);
        ASSERT(lat_id >= 0);
        ASSERT(lat_id < _lattices.size());

        // Increment the number of times we have seen this lattice
        Int const lat_id_ctr = ++lat_found[lat_id];

        // Get the lattice name
        String const lat_name =
            "Lattice_" + getASCIINumber(lat_id) + "_" + getASCIINumber(lat_id_ctr);
        LOG_DEBUG("Lattice name: ", lat_name);

        // Get the lattice offset (z direction)
        // The midplane is the location that the geometry was sampled at.
        Float const low_z = assembly.grid().divs(0)[izlat];
        Float const high_z = assembly.grid().divs(0)[izlat + 1];
        Float const lat_z = (low_z + high_z) / 2;

        // Get the lattice
        auto const & lattice = _lattices[lat_id];
        if (lattice.children().empty()) {
          logger::error("Lattice has no children");
          return core_soup;
        }

        // For each RTM
        Int const nyrtm = lattice.grid().numCells(1);
        Int const nxrtm = lattice.grid().numCells(0);
        for (Int iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (Int ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {

            PolytopeSoup rtm_soup;

            // Get the RTM ID
            auto const rtm_id = lattice.getChild(ixrtm, iyrtm);
            ASSERT(rtm_id >= 0);
            ASSERT(rtm_id < _rtms.size());

            // Increment the number of times we have seen this RTM
            Int const rtm_id_ctr = ++rtm_found[rtm_id];

            // Get the RTM name
            String const rtm_name =
                "RTM_" + getASCIINumber(rtm_id) + "_" + getASCIINumber(rtm_id_ctr);
            LOG_DEBUG("RTM name: ", rtm_name);

            // Get the RTM offset (lower left corner)
            Point2F const rtm_ll = lattice.grid().getBox(ixrtm, iyrtm).minima();

            // Get the rtm
            auto const & rtm = _rtms[rtm_id];
            if (rtm.children().empty()) {
              logger::error("RTM has no children");
              return core_soup;
            }

            // For each coarse cell
            Int const nycells = rtm.grid().numCells(1);
            Int const nxcells = rtm.grid().numCells(0);
            for (Int iycell = 0; iycell < nycells; ++iycell) {
              for (Int ixcell = 0; ixcell < nxcells; ++ixcell) {

                PolytopeSoup cell_soup;

                // Get the coarse cell ID
                auto const & cell_id = rtm.getChild(ixcell, iycell);
                ASSERT(cell_id >= 0);
                ASSERT(cell_id < _coarse_cells.size());

                // Increment the number of times we have seen this coarse cell
                Int const cell_id_ctr = ++cc_found[cell_id];

                // Get the coarse cell name
                String const cell_name = "Coarse_Cell_" + getASCIINumber(cell_id) + "_" +
                                         getASCIINumber(cell_id_ctr);
                LOG_DEBUG("Coarse cell name: ", cell_name);

                // Get the cell offset (lower left corner)
                Point2F const cell_ll = rtm.grid().getBox(ixcell, iycell).minima();

                // Get the coarse cell
                auto const & coarse_cell = _coarse_cells[cell_id];

                // Get the mesh type and id of the coarse cell.
                MeshType const mesh_type = coarse_cell.mesh_type;
                Int const mesh_id = coarse_cell.mesh_id;

                Vector<Float> mcls(coarse_cell.numFaces());

                switch (mesh_type) {
                case MeshType::Tri:
                  LOG_DEBUG("Mesh type: Tri");
                  cell_soup = _tris[mesh_id];
                  for (Int i = 0; i < mcls.size(); ++i) {
                    mcls[i] = _tris[mesh_id].getFace(i).meanChordLength();
                  }
                  break;
                case MeshType::Quad:
                  LOG_DEBUG("Mesh type: Quad");
                  cell_soup = _quads[mesh_id];
                  for (Int i = 0; i < mcls.size(); ++i) {
                    mcls[i] = _quads[mesh_id].getFace(i).meanChordLength();
                  }
                  break;
                case MeshType::QuadraticTri:
                  LOG_DEBUG("Mesh type: QuadraticTri");
                  cell_soup = _tri6s[mesh_id];
                  for (Int i = 0; i < mcls.size(); ++i) {
                    mcls[i] = _tri6s[mesh_id].getFace(i).meanChordLength();
                  }
                  break;
                case MeshType::QuadraticQuad:
                  LOG_DEBUG("Mesh type: QuadraticQuad");
                  cell_soup = _quad8s[mesh_id];
                  for (Int i = 0; i < mcls.size(); ++i) {
                    mcls[i] = _quad8s[mesh_id].getFace(i).meanChordLength();
                  }
                  break;
                default:
                  logger::error("Unsupported mesh type");
                  return core_soup;
                } // switch (mesh_type)

                // Translate the cell_soup to the correct location
                // global xyz offset
                Point2F const xy_offset = cell_ll + rtm_ll + asy_ll;
                Point3F const global_offset = Point3F(xy_offset[0], xy_offset[1], lat_z);
                cell_soup.translate(global_offset);

                Vector<Int> cell_ids(cell_soup.numElements());
                um2::iota(cell_ids.begin(), cell_ids.end(), 0);
                // Add the Coarse_Cell, RTM, Lattice, and Assembly elsets
                cell_soup.addElset(cell_name, cell_ids);
                cell_soup.addElset(rtm_name, cell_ids);
                cell_soup.addElset(lat_name, cell_ids);
                cell_soup.addElset(asy_name, cell_ids);

                // Add the material IDs
                Vector<Float> mat_ids(coarse_cell.material_ids.size());
                for (Int i = 0; i < mat_ids.size(); ++i) {
                  mat_ids[i] = static_cast<Float>(coarse_cell.material_ids[i]);
                }
                cell_soup.addElset("Material_ID", cell_ids, mat_ids);

                // Add the one-group cross sections
                Vector<Float> xsecs(coarse_cell.numFaces());
                for (Int i = 0; i < xsecs.size(); ++i) {
                  xsecs[i] = one_group_xs[static_cast<Int>(coarse_cell.material_ids[i])];
                }
                cell_soup.addElset("One_Group_XS", cell_ids, xsecs);

                // Add the mean chord lengths
                cell_soup.addElset("Mean_Chord_Length", cell_ids, mcls);

                // Add the Knudsen numbers
                // Kn = mean free path / characteristic length
                //    = 1 / (xs * mcl)
                for (Int i = 0; i < xsecs.size(); ++i) {
                  mcls[i] = 1 / (xsecs[i] * mcls[i]);
                }
                cell_soup.addElset("Knudsen_Number", cell_ids, mcls);

                cell_soup.sortElsets();
                rtm_soup += cell_soup;
              } // for (ixcell)
            } // for (iycell)
            lattice_soup += rtm_soup;
          } // for (ixrtm)
        } // for (iyrtm)
        assembly_soup += lattice_soup;
      } // for (izlat)
      core_soup += assembly_soup;
    } // for (ixasy)
  } // for (iyasy)
  if (core_soup.numElsets() != 0) {
    core_soup.sortElsets();
  }
  return core_soup;
} // operator PolytopeSoup

//==============================================================================
// writeXDMFFile
//==============================================================================

namespace
{

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
writeXDMFFile(String const & filepath, Model const & model,
              bool const write_knudsen_data = false, bool const write_xsec_data = false)
{
  LOG_INFO("Writing MPACT model to XDMF file: ", filepath);

  auto const & core = model.core();
  auto const & assemblies = model.assemblies();
  auto const & lattices = model.lattices();
  auto const & rtms = model.rtms();
  auto const & coarse_cells = model.coarseCells();
  auto const & materials = model.materials();
  auto const & tris = model.triMeshes();
  auto const & quads = model.quadMeshes();
  auto const & tri6s = model.tri6Meshes();
  auto const & quad8s = model.quad8Meshes();

  if (core.children().empty()) {
    logger::error("Core has no children");
    return;
  }

  // Process 1-group average material cross sections if requested
  Vector<Float> avg_xs_a(materials.size());
  Vector<Float> avg_xs_f(materials.size());
  Vector<Float> avg_xs_s(materials.size());
  Vector<Float> avg_xs_t(materials.size());
  if (write_knudsen_data || write_xsec_data) {
    for (Int imat = 0; imat < materials.size(); ++imat) {
      auto const & mat = materials[imat];
      mat.validateXSec();
      auto const xs = materials[imat].xsec().collapseTo1GroupAvg();
      avg_xs_a[imat] = xs.a()[0];
      avg_xs_f[imat] = xs.f()[0];
      avg_xs_s[imat] = xs.s()[0];
      avg_xs_t[imat] = xs.t(0);
    }
  }

  // Store a PolytopeSoup for each CoarseCell
  Vector<PolytopeSoup> coarse_cell_soups(coarse_cells.size());
  {
    Vector<Float> mcls;
    Vector<Int> cell_ids;
    Vector<Float> mat_ids;
    Vector<Float> xsecs;
    for (Int icc = 0; icc < coarse_cells.size(); ++icc) {
      auto & cell_soup = coarse_cell_soups[icc];
      auto const & coarse_cell = coarse_cells[icc];

      // Get the mesh type and id of the coarse cell.
      MeshType const mesh_type = coarse_cell.mesh_type;
      Int const mesh_id = coarse_cell.mesh_id;

      if (write_knudsen_data) {
        mcls.resize(coarse_cell.numFaces());
      }

      switch (mesh_type) {
      case MeshType::Tri:
        LOG_DEBUG("Mesh type: Tri");
        cell_soup = tris[mesh_id];
        if (write_knudsen_data) {
          for (Int i = 0; i < mcls.size(); ++i) {
            mcls[i] = tris[mesh_id].getFace(i).meanChordLength();
          }
        }
        break;
      case MeshType::Quad:
        LOG_DEBUG("Mesh type: Quad");
        cell_soup = quads[mesh_id];
        if (write_knudsen_data) {
          for (Int i = 0; i < mcls.size(); ++i) {
            mcls[i] = quads[mesh_id].getFace(i).meanChordLength();
          }
        }
        break;
      case MeshType::QuadraticTri:
        LOG_DEBUG("Mesh type: QuadraticTri");
        cell_soup = tri6s[mesh_id];
        if (write_knudsen_data) {
          for (Int i = 0; i < mcls.size(); ++i) {
            mcls[i] = tri6s[mesh_id].getFace(i).meanChordLength();
          }
        }
        break;
      case MeshType::QuadraticQuad:
        LOG_DEBUG("Mesh type: QuadraticQuad");
        cell_soup = quad8s[mesh_id];
        if (write_knudsen_data) {
          for (Int i = 0; i < mcls.size(); ++i) {
            mcls[i] = quad8s[mesh_id].getFace(i).meanChordLength();
          }
        }
        break;
      default:
        logger::error("Unsupported mesh type");
        return;
      } // switch (mesh_type)

      cell_ids.resize(cell_soup.numElements());
      um2::iota(cell_ids.begin(), cell_ids.end(), 0);

      // Add the material IDs
      mat_ids.resize(coarse_cell.material_ids.size());
      for (Int i = 0; i < mat_ids.size(); ++i) {
        mat_ids[i] = static_cast<Float>(coarse_cell.material_ids[i]);
      }
      cell_soup.addElset("Material_ID", cell_ids, mat_ids);

      // Add the one-group average cross sections if requested
      if (write_knudsen_data || write_xsec_data) {
        xsecs.resize(coarse_cell.numFaces());
        // Absorption
        for (Int i = 0; i < xsecs.size(); ++i) {
          xsecs[i] = avg_xs_a[static_cast<Int>(coarse_cell.material_ids[i])];
        }
        cell_soup.addElset("One_Group_Avg_XS_a", cell_ids, xsecs);
        // Fission
        for (Int i = 0; i < xsecs.size(); ++i) {
          xsecs[i] = avg_xs_f[static_cast<Int>(coarse_cell.material_ids[i])];
        }
        cell_soup.addElset("One_Group_Avg_XS_f", cell_ids, xsecs);
        // Scattering
        for (Int i = 0; i < xsecs.size(); ++i) {
          xsecs[i] = avg_xs_s[static_cast<Int>(coarse_cell.material_ids[i])];
        }
        cell_soup.addElset("One_Group_Avg_XS_s", cell_ids, xsecs);
        // Total
        for (Int i = 0; i < xsecs.size(); ++i) {
          xsecs[i] = avg_xs_t[static_cast<Int>(coarse_cell.material_ids[i])];
        }
        cell_soup.addElset("One_Group_Avg_XS_t", cell_ids, xsecs);
      }

      // Add the multigroup total cross sections if requested
      if (write_xsec_data) {
        Vector<Float> mg_xsec_data(coarse_cell.numFaces());
        Int const num_groups = materials[0].xsec().numGroups();
        // For each energy group:
        for (Int ig = 0; ig < num_groups; ++ig) {
          for (Int icell = 0; icell < mat_ids.size(); ++icell) {
            // NOLINTNEXTLINE(bugprone-signed-char-misuse,cert-str34-c)
            auto const mat_id = static_cast<Int>(coarse_cell.material_ids[icell]);
            auto const & xsec = materials[mat_id].xsec();
            mg_xsec_data[icell] = xsec.t(ig);
          }
          cell_soup.addElset("Group_" + getASCIINumber(ig) + "_Total_XS", cell_ids,
                             mg_xsec_data);
        }
      }

      if (write_knudsen_data) {
        // Add the mean chord lengths
        cell_soup.addElset("Mean_Chord_Length", cell_ids, mcls);

        // Add the Knudsen numbers
        // Kn = mean free path / characteristic length
        //    = 1 / (xs * mcl)
        for (Int i = 0; i < xsecs.size(); ++i) {
          mcls[i] = 1 / (xsecs[i] * mcls[i]);
        }
        cell_soup.addElset("Knudsen_Number", cell_ids, mcls);
      }

      cell_soup.sortElsets();
    } // for (icc)
  }

  // Setup HDF5 file
  // Get the h5 file name
  Int last_slash = filepath.find_last_of('/');
  if (last_slash == String::npos) {
    last_slash = 0;
  }

  // If there is no slash, the file name and path are the same
  // If there is a slash, the file name is everything after the last slash
  // and the path is everything before and including the last slash
  Int const h5filepath_end = last_slash == 0 ? 0 : last_slash + 1;
  ASSERT(h5filepath_end < filepath.size());
  // /some/path/foobar.xdmf -> foobar.h5
  String const h5filename =
      filepath.substr(h5filepath_end, filepath.size() - 5 - h5filepath_end) + ".h5";
  // /some/path/foobar.xdmf -> /some/path/
  String const h5filepath = filepath.substr(0, h5filepath_end);
  String const h5fullpath = h5filepath + h5filename;
  H5::H5File h5file(h5fullpath.data(), H5F_ACC_TRUNC);

  // Setup XML file
  pugi::xml_document xdoc;

  // XDMF root node
  pugi::xml_node xroot = xdoc.append_child("Xdmf");
  xroot.append_attribute("Version") = "3.0";

  // Domain node
  pugi::xml_node xdomain = xroot.append_child("Domain");

  // Write the material names as backup information
  Vector<String> material_names;
  for (auto const & material : materials) {
    material_names.push_back(material.getName());
  }
  pugi::xml_node xinfo = xdomain.append_child("Information");
  xinfo.append_attribute("Name") = "Materials";
  String mats;
  for (Int i = 0; i < material_names.size(); ++i) {
    mats += material_names[i];
    if (i + 1 < material_names.size()) {
      mats += ", ";
    }
  }
  xinfo.append_child(pugi::node_pcdata).set_value(mats.data());

  // Core
  String const name = h5filename.substr(0, h5filename.size() - 3);
  pugi::xml_node xcore_grid = xdomain.append_child("Grid");
  xcore_grid.append_attribute("Name") = name.data();
  xcore_grid.append_attribute("GridType") = "Tree";
  H5::Group const h5core_group = h5file.createGroup(name.data());
  String const h5core_grouppath = "/" + name;

  // Allocate counters for each assembly, lattice, etc.
  Vector<Int> asy_found(assemblies.size(), -1);
  Vector<Int> lat_found(lattices.size(), -1);
  Vector<Int> rtm_found(rtms.size(), -1);
  Vector<Int> cc_found(coarse_cells.size(), -1);

  Int const nyasy = core.grid().numCells(1);
  Int const nxasy = core.grid().numCells(0);

  // Core NX_by_NY
  pugi::xml_node xcore_info = xcore_grid.append_child("Information");
  xcore_info.append_attribute("Name") = "NX_by_NY";
  String const core_nx_by_ny = String(nxasy) + " x " + String(nyasy);
  xcore_info.append_child(pugi::node_pcdata).set_value(core_nx_by_ny.data());

  // For each assembly
  for (Int iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Int ixasy = 0; ixasy < nxasy; ++ixasy) {

      // Get the assembly ID
      auto const asy_id = core.getChild(ixasy, iyasy);
      ASSERT(asy_id >= 0);
      ASSERT(asy_id < assemblies.size());

      // Increment the number of times we have seen this assembly
      Int const asy_id_ctr = ++asy_found[asy_id];

      // Get the assembly name
      String const asy_name =
          "Assembly_" + getASCIINumber(asy_id) + "_" + getASCIINumber(asy_id_ctr);
      LOG_DEBUG("Assembly name: ", asy_name);

      // Create the assembly group
      pugi::xml_node xasy_grid = xcore_grid.append_child("Grid");
      xasy_grid.append_attribute("Name") = asy_name.data();
      xasy_grid.append_attribute("GridType") = "Tree";
      String const h5asy_grouppath = h5core_grouppath + "/" + asy_name;
      H5::Group const h5asy_group = h5file.createGroup(h5asy_grouppath.data());

      // Get the assembly offset (lower left corner)
      Point2F const asy_ll = core.grid().getBox(ixasy, iyasy).minima();

      // Get the assembly
      auto const & assembly = assemblies[asy_id];
      if (assembly.children().empty()) {
        logger::error("Assembly has no children");
        return;
      }

      Int const nzlat = assembly.grid().numCells(0);

      // Assembly NX_by_NY
      pugi::xml_node xasy_info = xasy_grid.append_child("Information");
      xasy_info.append_attribute("Name") = "NX_by_NY";
      String const asy_nx_by_ny = String(nzlat) + " x 1";
      xasy_info.append_child(pugi::node_pcdata).set_value(asy_nx_by_ny.data());

      // For each lattice
      for (Int izlat = 0; izlat < nzlat; ++izlat) {

        // Get the lattice ID
        auto const lat_id = assembly.getChild(izlat);
        ASSERT(lat_id >= 0);
        ASSERT(lat_id < lattices.size());

        // Increment the number of times we have seen this lattice
        Int const lat_id_ctr = ++lat_found[lat_id];

        // Get the lattice name
        String const lat_name =
            "Lattice_" + getASCIINumber(lat_id) + "_" + getASCIINumber(lat_id_ctr);
        LOG_DEBUG("Lattice name: ", lat_name);

        // Create the lattice group
        pugi::xml_node xlat_grid = xasy_grid.append_child("Grid");
        xlat_grid.append_attribute("Name") = lat_name.data();
        xlat_grid.append_attribute("GridType") = "Tree";
        String const h5lat_grouppath = h5asy_grouppath + "/" + lat_name;
        H5::Group const h5lat_group = h5file.createGroup(h5lat_grouppath.data());

        // Get the lattice offset (z direction)
        // The midplane is the location that the geometry was sampled at.
        Float const low_z = assembly.grid().divs(0)[izlat];
        Float const high_z = assembly.grid().divs(0)[izlat + 1];
        Float const lat_z = (low_z + high_z) / 2;

        // Add the lattice "Z" information
        pugi::xml_node xlat_info = xlat_grid.append_child("Information");
        xlat_info.append_attribute("Name") = "Z";
        String const lat_z_str =
            String(low_z) + ", " + String(lat_z) + ", " + String(high_z);
        xlat_info.append_child(pugi::node_pcdata).set_value(lat_z_str.data());

        // Get the lattice
        auto const & lattice = lattices[lat_id];
        if (lattice.children().empty()) {
          logger::error("Lattice has no children");
          return;
        }

        Int const nyrtm = lattice.grid().numCells(1);
        Int const nxrtm = lattice.grid().numCells(0);

        // Lattice NX_by_NY
        pugi::xml_node xlat_info2 = xlat_grid.append_child("Information");
        xlat_info2.append_attribute("Name") = "NX_by_NY";
        String const lat_nx_by_ny = String(nxrtm) + " x " + String(nyrtm);
        xlat_info2.append_child(pugi::node_pcdata).set_value(lat_nx_by_ny.data());

        // For each RTM
        for (Int iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (Int ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {

            // Get the RTM ID
            auto const rtm_id = lattice.getChild(ixrtm, iyrtm);
            ASSERT(rtm_id >= 0);
            ASSERT(rtm_id < rtms.size());

            // Increment the number of times we have seen this RTM
            Int const rtm_id_ctr = ++rtm_found[rtm_id];

            // Get the RTM name
            String const rtm_name =
                "RTM_" + getASCIINumber(rtm_id) + "_" + getASCIINumber(rtm_id_ctr);
            LOG_DEBUG("RTM name: ", rtm_name);

            // Create the RTM group
            pugi::xml_node xrtm_grid = xlat_grid.append_child("Grid");
            xrtm_grid.append_attribute("Name") = rtm_name.data();
            xrtm_grid.append_attribute("GridType") = "Tree";
            String const h5rtm_grouppath = h5lat_grouppath + "/" + rtm_name;
            H5::Group const h5rtm_group = h5file.createGroup(h5rtm_grouppath.data());

            // Get the RTM offset (lower left corner)
            Point2F const rtm_ll = lattice.grid().getBox(ixrtm, iyrtm).minima();

            // Get the rtm
            auto const & rtm = rtms[rtm_id];
            if (rtm.children().empty()) {
              logger::error("RTM has no children");
              return;
            }

            Int const nycells = rtm.grid().numCells(1);
            Int const nxcells = rtm.grid().numCells(0);

            // RTM NX_by_NY
            pugi::xml_node xrtm_info = xrtm_grid.append_child("Information");
            xrtm_info.append_attribute("Name") = "NX_by_NY";
            String const rtm_nx_by_ny = String(nxcells) + " x " + String(nycells);
            xrtm_info.append_child(pugi::node_pcdata).set_value(rtm_nx_by_ny.data());

            // For each coarse cell
            for (Int iycell = 0; iycell < nycells; ++iycell) {
              for (Int ixcell = 0; ixcell < nxcells; ++ixcell) {

                // Get the coarse cell ID
                auto const & cell_id = rtm.getChild(ixcell, iycell);
                ASSERT(cell_id >= 0);
                ASSERT(cell_id < coarse_cells.size());

                // Increment the number of times we have seen this coarse cell
                Int const cell_id_ctr = ++cc_found[cell_id];

                // Get the coarse cell name
                String const cell_name = "Coarse_Cell_" + getASCIINumber(cell_id) + "_" +
                                         getASCIINumber(cell_id_ctr);
                LOG_DEBUG("Coarse cell name: ", cell_name);

                // Get the cell offset (lower left corner)
                Point2F const cell_ll = rtm.grid().getBox(ixcell, iycell).minima();

                // Translate the cell_soup to the correct location
                // global xyz offset
                Point2F const xy_offset = cell_ll + rtm_ll + asy_ll;
                Point3F const global_offset = Point3F(xy_offset[0], xy_offset[1], lat_z);

                auto const & cell_soup = coarse_cell_soups[cell_id];

                writeXDMFUniformGrid(cell_name, xrtm_grid, h5file, h5filename,
                                     h5rtm_grouppath, cell_soup, global_offset);

              } // for (ixcell)
            } // for (iycell)
          } // for (ixrtm)
        } // for (iyrtm)
      } // for (izlat)
    } // for (ixasy)
  } // for (iyasy)

  // Write the XML file
  xdoc.save_file(filepath.data(), "  ");

  // Close the HDF5 file
  h5file.close();

  LOG_INFO("XDMF file written successfully");
} // writeXDMFFile

void
writeInputFile(String const & filename, Model const & model)
{

  // Strip the ending from filename
  Int const last_period = filename.find_last_of('.');
  String const base_filename = filename.substr(0, last_period);
  String const input_filename = base_filename + ".input";
  LOG_INFO("Writing MPACT input file: ", input_filename);

  // Open the file
  std::ofstream file(std::string(input_filename.data()));
  if (!file.is_open()) {
    logger::error("Failed to open file: ", input_filename);
    return;
  }

  // CASEID
  // strip everything before the last slash
  Int last_slash = filename.find_last_of('/');
  if (last_slash == String::npos) {
    last_slash = -1;
  }
  String const model_name = base_filename.substr(last_slash + 1);
  file << "CASEID MPACT_" << model_name.data() << "\n\n";

  // STATE
  // Assume we want power and flux
  file << "STATE\n";
  file << "  edit fsr_flux fsr_power\n\n";

  // MATERIAL
  file << "MATERIAL\n";
  Int const num_materials = model.materials().size();
  for (Int imat = 0; imat < num_materials; ++imat) {
    auto const & material = model.materials()[imat];
    auto const density = material.getDensity();
    auto const temperature = material.getTemperature();

    String const mat_info = "  mat " + String(imat + 1) + " <type> " + String(density) +
                            " g/cc " + String(temperature) + " K \\ ";
    Int const mat_info_size = mat_info.size() - 5; // Account for <type> -> 1 char

    file << "  ! " << material.getName().data() << "\n";
    file << mat_info.data();

    // Write the ZAID and number density of each isotope
    auto const & zaids = material.zaids();
    auto const & densities = material.numDensities();
    if (!zaids.empty()) {
      ASSERT(zaids.size() == densities.size());
      file << zaids[0] << " " << densities[0] << '\n';
      for (Int i = 1; i < zaids.size(); ++i) {
        // Insert the spaces
        for (Int j = 0; j < mat_info_size; ++j) {
          file << ' ';
        }
        file << zaids[i] << " " << densities[i] << '\n';
      }
    }
  } // for (imat)

  // GEOM
  file << "GEOM\n";
  file << "  um2_core\n";
  file << "    " << filename.data() << "\n\n";

  // XSEC
  file << "XSEC\n";
  file << "  xslib ORNL mpact51g_71_v4.2m5_12062016_sph.fmt\n\n";

  // OPTION
  // Get the mean chord length of the faces
  auto mcls = model.getMeanChordLengths();
  std::sort(mcls.begin(), mcls.end());
  Int const num_faces = mcls.size();
  // 90% of faces largest, means the bottom 10% of faces are not representative
  Int const num_faces_rep = static_cast<Int>((1 - 0.9) * num_faces);
  Float const mcl = mcls[num_faces_rep]; // The representative mean chord length
  file << "OPTION\n";
  file << "  bound_cond <west_bc> <north_bc> <east_bc> <south_bc> <top_bc> <bottom_bc>\n";
  file << "  solver 1 2\n";
  file << "  ! ray spacing is set such that the expected number of intersections in the "
          "largest\n";
  file << "  ! 90% of faces is 4+ for any given direction. (mean chord length / 4)\n";
  file << "  ray " << mcl / 4 << " CHEBYSHEV-YAMAMOTO 16 3\n";
  file << "  parallel 1 1 1 1\n";
  file << "  conv_crit 2*1.e-6\n";
  file << "  iter_lim 2000 1 1\n";
  file << "  vis_edits T T\n";
  file << "  cmfd A mgnode F 0.0 1.0 150 F";

  // Close the file
  file.close();

} // writeInputFile

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
writeMaterialVolumes(String const & filename, Model const & model)
{

  // Strip the ending from filename
  Int const last_period = filename.find_last_of('.');
  String const base_filename = filename.substr(0, last_period);
  String const volumes_filename = base_filename + ".volumes";
  LOG_INFO("Writing MPACT material volumes file: ", volumes_filename);

  // Open the file
  std::ofstream file(std::string(volumes_filename.data()));
  if (!file.is_open()) {
    logger::error("Failed to open file: ", volumes_filename);
    return;
  }

  auto const & core = model.core();
  auto const & lattices = model.lattices();
  auto const & rtms = model.rtms();
  auto const & coarse_cells = model.coarseCells();
  auto const & materials = model.materials();
  auto const & tris = model.triMeshes();
  auto const & quads = model.quadMeshes();
  auto const & tri6s = model.tri6Meshes();
  auto const & quad8s = model.quad8Meshes();

  Int const nmats = materials.size();
  Vector<Vector<Float>> volumes(nmats);
  // For each assembly in the core
  for (auto const & asy_id : core.children()) {
    // Get the assembly and determine the number of lattices
    auto const & assembly = model.getAssembly(asy_id);
    auto const num_lats = assembly.children().size();
    // For each of the lattices in the assembly
    for (Int ilat = 0; ilat < num_lats; ++ilat) {
      auto const lat_id = assembly.getChild(ilat);
      // Get the dz of the lattice
      Float const low_z = assembly.grid().divs(0)[ilat];
      Float const high_z = assembly.grid().divs(0)[ilat + 1];
      Float const dz = high_z - low_z;
      // For each RTM in the lattice
      for (auto const & rtm_id : lattices[lat_id].children()) {
        // For each coarse cell in the RTM
        for (auto const & cc_id : rtms[rtm_id].children()) {
          // Get the coarse cell
          auto const & cc = coarse_cells[cc_id];
          // Based on the mesh type, find the area of each face
          // NOLINTBEGIN(bugprone-signed-char-misuse,cert-str34-c)
          switch (cc.mesh_type) {
          case MeshType::Tri: {
            auto const & mesh = tris[cc.mesh_id];
            Int const num_faces = mesh.numFaces();
            for (Int i = 0; i < num_faces; ++i) {
              auto const mat_id = static_cast<Int>(cc.material_ids[i]);
              volumes[mat_id].emplace_back(mesh.getFace(i).area() * dz);
            }
            break;
          }
          case MeshType::Quad: {
            auto const & mesh = quads[cc.mesh_id];
            Int const num_faces = mesh.numFaces();
            for (Int i = 0; i < num_faces; ++i) {
              auto const mat_id = static_cast<Int>(cc.material_ids[i]);
              volumes[mat_id].emplace_back(mesh.getFace(i).area() * dz);
            }
            break;
          }
          case MeshType::QuadraticTri: {
            auto const & mesh = tri6s[cc.mesh_id];
            Int const num_faces = mesh.numFaces();
            for (Int i = 0; i < num_faces; ++i) {
              auto const mat_id = static_cast<Int>(cc.material_ids[i]);
              volumes[mat_id].emplace_back(mesh.getFace(i).area() * dz);
            }
            break;
          }
          case MeshType::QuadraticQuad: {
            auto const & mesh = quad8s[cc.mesh_id];
            Int const num_faces = mesh.numFaces();
            for (Int i = 0; i < num_faces; ++i) {
              auto const mat_id = static_cast<Int>(cc.material_ids[i]);
              volumes[mat_id].emplace_back(mesh.getFace(i).area() * dz);
            }
            break;
          }
          default:
            logger::error("Unsupported mesh type");
          }
          // NOLINTEND(bugprone-signed-char-misuse,cert-str34-c)
        }
      }
    }
  }

  // Reduce the volumes to the total volume of each material
  Vector<Float> total_volumes(nmats, 0);
  for (Int imat = 0; imat < nmats; ++imat) {
    // Sort and use the sum function for minimal floating point error
    // (sum uses pairwise summation)
    std::sort(volumes[imat].begin(), volumes[imat].end());
    total_volumes[imat] = um2::sum(volumes[imat].cbegin(), volumes[imat].cend());
  }

  Int p = 6;
  if constexpr (std::same_as<Float, double>) {
    p = 15;
  }

  for (Int imat = 0; imat < nmats; ++imat) {
    file << materials[imat].getName().data() << " " << std::setprecision(p)
         << total_volumes[imat] << '\n';
  }

  // Close the file
  file.close();

} // writeMaterialVolumes

void
writeCrossSections(String const & filename, Model const & model)
{

  // Strip the ending from filename
  Int const last_period = filename.find_last_of('.');
  String const base_filename = filename.substr(0, last_period);
  String const xs_filename = base_filename + ".xs";
  LOG_INFO("Writing MPACT material cross sections file: ", xs_filename);

  // Open the file
  std::ofstream file(std::string(xs_filename.data()));
  if (!file.is_open()) {
    logger::error("Failed to open file: ", xs_filename);
    return;
  }

  auto const & materials = model.materials();
  Int const nmats = materials.size();
  Int const num_groups = materials[0].xsec().numGroups();
  Int precision = 6;
  if constexpr (std::same_as<Float, double>) {
    precision = 15;
  }
  // Write number of materials and number of groups
  file << nmats << " " << num_groups << '\n';
  for (auto const & material : materials) {
    // Write the material name
    file << material.getName().data() << '\n';
    // Write the cross sections
    auto const & xsec = material.xsec();
    ASSERT(xsec.numGroups() == num_groups);
    for (Int ig = 0; ig < num_groups; ++ig) {
      file << std::setprecision(precision) << xsec.a()[ig] << ' ' << xsec.f()[ig] << ' '
           << xsec.nuf()[ig] << ' ' << xsec.tr()[ig] << ' ' << xsec.s()[ig] << '\n';
    }
  }

  // Close the file
  file.close();
} // writeCrossSections

} // namespace

//==============================================================================
// write
//==============================================================================

void
Model::write(String const & filename, bool const write_knudsen_data,
             bool const write_xsec_data) const
{
  if (filename.ends_with(".xdmf")) {
    writeXDMFFile(filename, *this, write_knudsen_data, write_xsec_data);
  } else {
    logger::error("Unsupported file format.");
  }
  // Write the MPACT input file
  writeInputFile(filename, *this);
  // Write the volumes of each material
  writeMaterialVolumes(filename, *this);
  // Write the cross sections of each material
  writeCrossSections(filename, *this);
}

//==============================================================================
// readXDMFFile
//==============================================================================

namespace
{

void
getNXbyNY(pugi::xml_node const & xgrid, Int & nx, Int & ny)
{
  pugi::xml_node const xinfo = xgrid.child("Information");
  pugi::xml_attribute const xname = xinfo.attribute("Name");
  if (strcmp("NX_by_NY", xname.value()) != 0) {
    logger::error("XDMF XML information name is not NX_by_NY");
    nx = -1;
    ny = -1;
    return;
  }
  // String of the form "nx x ny"
  String const nx_by_ny = xinfo.child_value();
  StringView sv(nx_by_ny);
  StringView const nx_token = sv.getTokenAndShrink();
  sv.remove_prefix(2);
  char * end = nullptr;
  nx = strto<Int>(nx_token.data(), &end);
  ASSERT(end != nullptr);
  end = nullptr;
  ny = strto<Int>(sv.data(), &end);
  ASSERT(end != nullptr);
}

// y
// ^
// | { { 7, 8, 9},
// |   { 4, 5, 6}
// |   { 1, 2, 3} }
// |
// +---------> x
// flat | value | (i, j)
// -----+-------+-------
// 0    | 1     | (0, 2)
// 1    | 2     | (1, 2)
// 2    | 3     | (2, 2)
// 3    | 4     | (0, 1)
// 4    | 5     | (1, 1)
// 5    | 6     | (2, 1)
// 6    | 7     | (0, 0)
// 7    | 8     | (1, 0)
// 8    | 9     | (2, 0)
// Map a flat lattice: flat_idx to 2D lattice: (i, j)
PURE auto
mapFlatIndexToLattice2D(Int const flat_idx, Int const nx, Int const ny) -> Vec2I
{
  ASSERT(nx > 0);
  ASSERT(ny > 0);
  Int const i = flat_idx % nx;
  Int const j = ny - (flat_idx / nx) - 1;
  ASSERT(0 <= i);
  ASSERT(i < nx);
  ASSERT(0 <= j);
  ASSERT(j < ny);
  return {i, j};
}

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
readXDMFFile(String const & filename, Model & model)
{
  LOG_INFO("Reading MPACT model from XDMF file: ", filename);

  // Open HDF5 file
  Int last_slash = filename.find_last_of('/');
  if (last_slash == String::npos) {
    last_slash = 0;
  }
  Int const h5filepath_end = last_slash == 0 ? 0 : last_slash + 1;
  ASSERT(h5filepath_end < filename.size());
  String const h5filename =
      filename.substr(h5filepath_end, filename.size() - 5 - h5filepath_end) + ".h5";
  String const h5filepath = filename.substr(0, h5filepath_end);
  String const h5fullpath = h5filepath + h5filename;
  H5::H5File const h5file(h5fullpath.data(), H5F_ACC_RDONLY);

  // Setup XML file
  pugi::xml_document xdoc;
  pugi::xml_parse_result const result = xdoc.load_file(filename.data());
  if (!result) {
    logger::error("XDMF XML parse error: ", result.description(),
                  ", character pos= ", result.offset);
    return;
  }
  pugi::xml_node const xroot = xdoc.child("Xdmf");
  if (strcmp("Xdmf", xroot.name()) != 0) {
    logger::error("XDMF XML root node is not Xdmf");
    return;
  }
  pugi::xml_node const xdomain = xroot.child("Domain");
  if (strcmp("Domain", xdomain.name()) != 0) {
    logger::error("XDMF XML domain node is not Domain");
    return;
  }

  // Get the material names
  pugi::xml_node const xinfo = xdomain.child("Information");
  if (strcmp("Information", xinfo.name()) != 0) {
    logger::error("XDMF XML information node is not Information");
    return;
  }
  // Get the "Name" attribute
  pugi::xml_attribute const xname = xinfo.attribute("Name");
  if (strcmp("Materials", xname.value()) != 0) {
    logger::error("XDMF XML information name is not Materials");
    return;
  }

  // Add the materials
  //---------------------------------------------------------------------------
  {
    // Get the material names
    String const mats = xinfo.child_value();
    StringView mats_view(mats);
    while (mats_view.find_first_of(',') != StringView::npos) {
      StringView token = mats_view.getTokenAndShrink(',');
      token.removeLeadingSpaces();
      Material mat;
      mat.setName(String(token));
      model.addMaterial(mat, /*validate=*/false);
    }
    // Add the final material
    mats_view.removeLeadingSpaces();
    Material mat;
    mat.setName(String(mats_view));
    model.addMaterial(mat, /*validate=*/false);
  }

  //============================================================================
  // Algorithm for populating the model
  //============================================================================
  //
  // Constraints:
  // - We want to use the makeCore, makeAssembly, makeLattice,
  // makeRTM, and makeCoarseCell methods to create the model. These functions take
  // children IDs as arguments.
  // - We have to create ID 1 before ID 2, ID 2 before ID 3, etc.
  // We have to construct the model in a bottom-up fashion, i.e. we have to create
  // the coarse cells before we can create the RTMs, etc.
  // - We want to avoid making multiple passes over the XDMF file.
  //
  // Algorithm:
  // ========================================================================
  // Get the core node
  // Get the NX by NY size of the core
  // Allocate core_assembly_ids to be NX by NY
  // Loop over all assemblies
  //   Get the assembly node
  //   Extract the assembly ID from the name
  //   Write the assembly ID to core_assembly_ids
  //   If the assembly ID is not in assembly_ids
  //     Insert the ID to assembly_ids
  //     Get NX by NY size of the assembly (N = 1 always)
  //     Allocate assembly_lattice_ids to M
  //     Allocate assembly_lattice_zs to M + 1
  //     Loop over all lattices
  //       Get the lattice node
  //       Extract the lattice ID from the name
  //       Write the lattice ID to assembly_lattice_ids
  //       Get the Z positions of the lattice
  //       If this is the first lattice write the top and bottom Z positions to
  //       assembly_lattice_zs Else write the top Z position to assembly_lattice_zs
  //       If the lattice ID is not in lattice_ids
  //         Insert the ID to lattice_ids
  //         Get the NX by NY size of the lattice
  //         Allocate lattice_rtm_ids to NX by NY
  //         Loop over all RTMs
  //           Get the RTM node
  //           Extract the RTM ID from the name
  //           Write the RTM ID to lattice_rtm_ids
  //           If the RTM ID is not in rtm_ids
  //             Insert the ID to rtm_ids
  //             Get the NX by NY size of the RTM
  //             Allocate rtm_coarse_cell_ids to NX by NY
  //             Loop over all coarse cells
  //               Get the coarse cell node
  //               Extract the coarse cell ID from the name
  //               Write the coarse cell ID to rtm_coarse_cell_ids
  //               If the coarse cell ID is not in coarse_cell_ids
  //                 Insert the ID to coarse_cell_ids
  //                 Read the mesh into a PolytopeSoup object using readXDMFUniformGrid
  //                 Set the coarse cell mesh type, mesh id, and material IDs
  //                 Create the mesh
  //                 Use the bounding box of the mesh to set the coarse cell xy_extent
  //
  // Now that we have all the IDs we can create the model
  // For each coarse cell,
  //   Use makeCoarseCell to create the coarse cell
  //   Add the mesh to the model
  //   Adjust the mesh id of the coarse cell to be the index of the mesh in the model
  // For each RTM,
  //   Use makeRTM to create the RTM
  // For each lattice,
  //   Use makeLattice to create the lattice
  // For each assembly,
  //   Use makeAssembly to create the assembly
  // Use makeCore to create the core

  // 2D map of assembly IDs in the core
  Vector<Vector<Int>> core_assembly_ids;
  // 1D map of lattice IDs in each assembly
  Vector<Vector<Int>> assembly_lattice_ids;
  // 1D map of Z positions of each lattice in each assembly
  Vector<Vector<Float>> assembly_lattice_zs;
  // 2D map of RTM IDs in each lattice
  Vector<Vector<Vector<Int>>> lattice_rtm_ids;
  // 2D map of coarse cell IDs in each RTM
  Vector<Vector<Vector<Int>>> rtm_coarse_cell_ids;

  Vector<Int> assembly_ids;    // IDs of all unique assemblies
  Vector<Int> lattice_ids;     // IDs of all unique lattices
  Vector<Int> rtm_ids;         // IDs of all unique RTMs
  Vector<Int> coarse_cell_ids; // IDs of all unique coarse cells

  Vector<Pair<MeshType, Int>> mesh_types_ids;
  Int tris_count = 0;
  Int quads_count = 0;
  Int tri6s_count = 0;
  Int quad8s_count = 0;
  Vector<Vec2F> xy_extents;
  Vector<Vector<MatID>> coarse_cell_material_ids;

  // Get the core node
  pugi::xml_node const xcore = xdomain.child("Grid");
  if (strcmp("Grid", xcore.name()) != 0) {
    logger::error("XDMF XML core node is not Grid");
    return;
  }
  if (strcmp("Tree", xcore.attribute("GridType").value()) != 0) {
    logger::error("Expected core GridType=Tree");
    return;
  }
  // Get the nx by ny size of the core
  Int core_nx = 0;
  Int core_ny = 0;
  getNXbyNY(xcore, core_nx, core_ny);
  ASSERT(core_nx > 0);
  ASSERT(core_ny > 0);
  LOG_DEBUG("Core NX_by_NY: ", core_nx, " x ", core_ny);

  // Allocate core_assembly_ids
  core_assembly_ids.resize(core_ny);
  for (Int iy = 0; iy < core_ny; ++iy) {
    core_assembly_ids[iy].resize(core_nx);
    um2::fill(core_assembly_ids[iy].begin(), core_assembly_ids[iy].end(), -1);
  }

  char * end = nullptr;

  Int assembly_count = 0;
  // Loop over all assemblies
  for (auto const & assembly_node : xcore.children("Grid")) {
    // Extract the assembly ID from the name
    // Of the form Assembly_XXXXX_YYYYY, where XXXXX is the assembly ID
    String const assembly_name = assembly_node.attribute("Name").value();
    String const assembly_id_str = assembly_name.substr(9, 5);
    Int const assembly_id = strto<Int>(assembly_id_str.data(), &end);
    ASSERT(end != nullptr);
    end = nullptr;

    // Write the assembly ID to core_assembly_ids
    auto const core_ij = mapFlatIndexToLattice2D(assembly_count, core_nx, core_ny);
    core_assembly_ids[core_ij[1]][core_ij[0]] = assembly_id;
    ++assembly_count;

    // If the assembly ID is not in assembly_ids, we need to find the ids of the
    // lattices, RTMs, and coarse cells in the assembly
    bool asy_id_found = false;
    for (auto const & asy_id : assembly_ids) {
      if (asy_id == assembly_id) {
        asy_id_found = true;
        break;
      }
    }
    if (asy_id_found) {
      continue;
    }

    LOG_DEBUG("New assembly ID: ", assembly_id);
    assembly_ids.emplace_back(assembly_id);

    // Get NX by NY size of the assembly (NY = 1 always)
    Int assembly_nx = 0;
    Int assembly_ny = 0;
    getNXbyNY(assembly_node, assembly_nx, assembly_ny);
    ASSERT(assembly_nx > 0);
    if (assembly_ny != 1) {
      logger::error("Assembly NX_by_NY is not NX x 1");
      return;
    }
    LOG_DEBUG("Assembly NX_by_NY: ", assembly_nx, " x ", assembly_ny);

    // Allocate assembly_lattice_ids to NX
    assembly_lattice_ids.emplace_back(assembly_nx);

    // Allocate assembly_lattice_zs to NX + 1
    assembly_lattice_zs.emplace_back(assembly_nx + 1);

    // Loop over all lattices
    Int lattice_count = 0;
    for (auto const & lattice_node : assembly_node.children("Grid")) {
      // Extract the lattice ID from the name
      // Of the form Lattice_XXXXX_YYYYY, where XXXXX is the lattice ID
      String const lattice_name = lattice_node.attribute("Name").value();
      String const lattice_id_str = lattice_name.substr(8, 5);
      Int const lattice_id = strto<Int>(lattice_id_str.data(), &end);
      ASSERT(end != nullptr);
      end = nullptr;

      // Write the lattice ID to assembly_lattice_ids
      assembly_lattice_ids.back()[lattice_count] = lattice_id;

      // Get the Z positions of the lattice
      auto low_z = infDistance<Float>();
      auto high_z = -infDistance<Float>();
      pugi::xml_node const lattice_info = lattice_node.child("Information");
      if (strcmp("Information", lattice_info.name()) != 0) {
        logger::error("XDMF XML lattice information node is not Information");
        return;
      }
      if (strcmp("Z", lattice_info.attribute("Name").value()) != 0) {
        logger::error("XDMF XML lattice information name is not Z");
        return;
      }
      String const z_str = lattice_info.child_value();
      StringView z_view(z_str);
      StringView const token = z_view.getTokenAndShrink(',');
      low_z = strto<Float>(token.data(), &end);
      ASSERT(end != nullptr);
      end = nullptr;
      z_view.getTokenAndShrink(',');
      high_z = strto<Float>(z_view.data(), &end);
      ASSERT(end != nullptr);
      end = nullptr;
      ASSERT(low_z < high_z);
      LOG_DEBUG("Lattice Z: ", low_z, ", ", high_z);

      // If this is the first lattice, write the top and bottom Z positions to
      // assembly_lattice_zs else write the top Z position to assembly_lattice_zs
      if (lattice_count == 0) {
        assembly_lattice_zs.back()[lattice_count] = low_z;
      }
      assembly_lattice_zs.back()[lattice_count + 1] = high_z;
      ++lattice_count;

      // If the lattice ID is not in lattice_ids, we need to find the ids of the
      // RTMs and coarse cells in the lattice
      bool lat_id_found = false;
      for (auto const & lat_id : lattice_ids) {
        if (lat_id == lattice_id) {
          lat_id_found = true;
          break;
        }
      }
      if (lat_id_found) {
        continue;
      }

      LOG_DEBUG("New lattice ID: ", lattice_id);
      lattice_ids.emplace_back(assembly_id);

      // Get NX by NY size of the lattice
      Int lattice_nx = 0;
      Int lattice_ny = 0;
      // Do this one manually
      {
        pugi::xml_node const xlat_nxny =
            lattice_node.child("Information").next_sibling("Information");
        pugi::xml_attribute const xlatname = xlat_nxny.attribute("Name");
        if (strcmp("NX_by_NY", xlatname.value()) != 0) {
          logger::error("XDMF XML information name is not NX_by_NY");
          return;
        }
        // String of the form "nx x ny"
        String const nx_by_ny = xlat_nxny.child_value();
        StringView sv(nx_by_ny);
        StringView const nx_token = sv.getTokenAndShrink();
        sv.remove_prefix(2);
        lattice_nx = strto<Int>(nx_token.data(), &end);
        ASSERT(end != nullptr);
        end = nullptr;
        lattice_ny = strto<Int>(sv.data(), &end);
        ASSERT(end != nullptr);
        end = nullptr;
      }
      ASSERT(lattice_nx > 0);
      ASSERT(lattice_ny > 0);
      LOG_DEBUG("Lattice NX_by_NY: ", lattice_nx, " x ", lattice_ny);

      // Allocate lattice_rtm_ids to NX by NY
      lattice_rtm_ids.emplace_back(lattice_ny);
      for (Int iy = 0; iy < lattice_ny; ++iy) {
        auto & row = lattice_rtm_ids.back()[iy];
        row.resize(lattice_nx);
        um2::fill(row.begin(), row.end(), -1);
      }

      // Loop over all RTMs
      Int rtm_count = 0;
      for (auto const & rtm_node : lattice_node.children("Grid")) {
        // Extract the RTM ID from the name
        // Of the form RTM_XXXXX_YYYYY, where XXXXX is the RTM ID
        String const rtm_name = rtm_node.attribute("Name").value();
        String const rtm_id_str = rtm_name.substr(4, 5);
        Int const rtm_id = strto<Int>(rtm_id_str.data(), &end);
        ASSERT(end != nullptr);
        end = nullptr;

        // Write the RTM ID to lattice_rtm_ids
        auto const lattice_ij =
            mapFlatIndexToLattice2D(rtm_count, lattice_nx, lattice_ny);
        lattice_rtm_ids.back()[lattice_ij[1]][lattice_ij[0]] = rtm_id;
        ++rtm_count;

        // If the RTM ID is not in rtm_ids, we need to find the ids of the
        // coarse cells in the RTM
        bool rtm_id_found = false;
        for (auto const & rtm_idv : rtm_ids) {
          if (rtm_id == rtm_idv) {
            rtm_id_found = true;
            break;
          }
        }
        if (rtm_id_found) {
          continue;
        }

        LOG_DEBUG("New RTM ID: ", rtm_id);
        rtm_ids.emplace_back(rtm_id);

        // Get NX by NY size of the RTM
        Int rtm_nx = 0;
        Int rtm_ny = 0;
        getNXbyNY(rtm_node, rtm_nx, rtm_ny);
        ASSERT(rtm_nx > 0);
        ASSERT(rtm_ny > 0);
        LOG_DEBUG("RTM NX_by_NY: ", rtm_nx, " x ", rtm_ny);

        // Allocate rtm_coarse_cell_ids to NX by NY
        rtm_coarse_cell_ids.emplace_back(rtm_ny);
        for (Int iy = 0; iy < rtm_ny; ++iy) {
          auto & row = rtm_coarse_cell_ids.back()[iy];
          row.resize(rtm_nx);
          um2::fill(row.begin(), row.end(), -1);
        }

        // Loop over all coarse cells
        Int coarse_cell_count = 0;
        for (auto const & coarse_cell_node : rtm_node.children("Grid")) {
          // Extract the coarse cell ID from the name
          // Of the form Coarse_Cell_XXXXX_YYYYY, where XXXXX is the coarse cell ID
          String const coarse_cell_name = coarse_cell_node.attribute("Name").value();
          String const coarse_cell_id_str = coarse_cell_name.substr(12, 5);
          Int const coarse_cell_id = strto<Int>(coarse_cell_id_str.data(), &end);
          ASSERT(end != nullptr);
          end = nullptr;

          // Write the coarse cell ID to rtm_coarse_cell_ids
          auto const rtm_ij = mapFlatIndexToLattice2D(coarse_cell_count, rtm_nx, rtm_ny);
          rtm_coarse_cell_ids.back()[rtm_ij[1]][rtm_ij[0]] = coarse_cell_id;
          ++coarse_cell_count;

          // If the coarse cell ID is not in coarse_cell_ids, we need to read the mesh
          // and create the coarse cell
          bool coarse_cell_id_found = false;
          for (auto const & cc_id : coarse_cell_ids) {
            if (cc_id == coarse_cell_id) {
              coarse_cell_id_found = true;
              break;
            }
          }
          if (coarse_cell_id_found) {
            continue;
          }
          LOG_DEBUG("New coarse cell ID: ", coarse_cell_id);
          coarse_cell_ids.emplace_back(coarse_cell_id);

          // Read the mesh into a PolytopeSoup using readXDMFUniformGrid
          PolytopeSoup soup;
          readXDMFUniformGrid(coarse_cell_node, h5file, h5filename, soup);

          // Determine the mesh type
          MeshType const mesh_type = getMeshType(soup.getElemTypes());
          ASSERT(mesh_type != MeshType::Invalid);
          ASSERT(mesh_type != MeshType::TriQuad);
          ASSERT(mesh_type != MeshType::QuadraticTriQuad);

          // Create the FVM and get the ID
          switch (mesh_type) {
          case MeshType::Tri: {
            TriFVM mesh(soup);
            auto const bb = mesh.boundingBox();
            auto const minima = bb.minima();
            for (auto & vert : mesh.vertices()) {
              vert -= minima;
            }
            xy_extents.emplace_back(bb.extents());
            model.addTriMesh(mesh);
            mesh_types_ids.emplace_back(mesh_type, tris_count);
            ++tris_count;
          } break;
          case MeshType::Quad: {
            QuadFVM mesh(soup);
            auto const bb = mesh.boundingBox();
            auto const minima = bb.minima();
            for (auto & vert : mesh.vertices()) {
              vert -= minima;
            }
            xy_extents.emplace_back(bb.extents());
            model.addQuadMesh(mesh);
            mesh_types_ids.emplace_back(mesh_type, quads_count);
            ++quads_count;
          } break;
          case MeshType::QuadraticTri: {
            Tri6FVM mesh(soup);
            auto const bb = mesh.boundingBox();
            auto const minima = bb.minima();
            for (auto & vert : mesh.vertices()) {
              vert -= minima;
            }
            xy_extents.emplace_back(bb.extents());
            model.addTri6Mesh(mesh);
            mesh_types_ids.emplace_back(mesh_type, tri6s_count);
            ++tri6s_count;
          } break;
          case MeshType::QuadraticQuad: {
            Quad8FVM mesh(soup);
            auto const bb = mesh.boundingBox();
            auto const minima = bb.minima();
            for (auto & vert : mesh.vertices()) {
              vert -= minima;
            }
            xy_extents.emplace_back(bb.extents());
            model.addQuad8Mesh(mesh);
            mesh_types_ids.emplace_back(mesh_type, quad8s_count);
            ++quad8s_count;
          } break;
          default:
            logger::error("Unsupported mesh type");
            return;
          }

          // Get the material IDs (an elset as Floats)
          Vector<Int> ids;
          Vector<Float> data;
          soup.getElset("Material_ID", ids, data);
          ASSERT(ids.size() == soup.numElements());
          ASSERT(data.size() == soup.numElements());
          for (Int i = 0; i < ids.size(); ++i) {
            ASSERT(ids[i] == i);
          }
          coarse_cell_material_ids.emplace_back(data.size());
          for (Int i = 0; i < data.size(); ++i) {
            coarse_cell_material_ids.back()[i] = static_cast<MatID>(data[i]);
          }

          // // If there is multigroup xsec data present, we can use it
          // // to populate the material cross sections. First, check if we
          // // have already populated the material cross sections.

          // bool have_all_xsec_data = true;
          // for (auto const & material : model.materials()) {
          //   if (!material.hasXSec()) {
          //     have_all_xsec_data = false;
          //     break;
          //   }
          // }

          // if (have_all_xsec_data) {
          //   continue;
          // }

          // // Check if the multigroup xsec data is present
          // Int num_groups = 0;
          // for (auto const & elset_name : soup.elsetNames()) {
          //   if (elset_name.starts_with("Group_") &&
          //       elset_name.ends_with("_Total_XS")) {
          //     ++num_groups;
          //   }
          // }

          // if (num_groups == 0) {
          //   continue;
          // }

          // // Get the material indices that don't have cross section data
          // Int const num_global_mats = model.materials().size();
          // Vector<MatID> missing_mat_indices;
          // missing_mat_indices.reserve(num_global_mats);
          // for (MatID imat = 0; imat < static_cast<MatID>(num_global_mats); ++imat) {
          //   auto & material = model.materials()[static_cast<Int>(imat)];
          //   if (!material.hasXSec()) {
          //     missing_mat_indices.emplace_back(imat);
          //     material.xsec().t().resize(num_groups);
          //     // Set the values to -infDistance<Float>() to indicate that the data is
          //     missing um2::fill(material.xsec().t().begin(), material.xsec().t().end(),
          //     -infDistance<Float>());
          //   }
          // }

          // // For each material that doesn't have cross section data, check to see if
          // // there is a cell with that material ID.
          // auto const & cc_material_ids = coarse_cell_material_ids.back();
          // for (auto const imat : missing_mat_indices) {
          //   Int cell_idx = -1;
          //   for (Int icell = 0; icell < cc_material_ids.size(); ++icell) {
          //     auto const mat_id = cc_material_ids[icell];
          //     if (imat == mat_id) {
          //       cell_idx = icell;
          //       break;
          //     }
          //   }
          //   // No cell with this material ID
          //   if (cell_idx == -1) {
          //     continue;
          //   }
          //   LOG_INFO("Found cross section data for material ID: ",
          //   static_cast<Int>(imat));

          //   // We have found a cell with the material ID. Get the elset data for each
          //   // energy group and emplace back into the corresponding material.
          //   auto & material = model.materials()[static_cast<Int>(imat)];

          //   // Get the cross section data
          //   Vector<Int> elset_ids;
          //   Vector<Float> elset_data;
          //   for (Int igroup = 0; igroup < num_groups; ++igroup) {
          //     String const elset_name = "Group_" + getASCIINumber(igroup) +
          //     "_Total_XS"; soup.getElset(elset_name, elset_ids, elset_data);
          //     ASSERT(elset_ids.size() == soup.numElements());
          //     ASSERT(elset_data.size() == soup.numElements());
          //     material.xsec().t(igroup) = elset_data[cell_idx];
          //     elset_ids.clear();
          //     elset_data.clear();
          //   }

          //   material.xsec().isMacro() = true;
          //   material.validateXSec();

          // } // Missing material indices loop
        } // Coarse cell loop
      } // RTM loop
    } // Lattice loop
  } // Assembly loop

  // If we have materials without cross section data, warn the user
  for (auto const & material : model.materials()) {
    if (!material.hasXSec()) {
      LOG_WARN("Material ", material.getName(), " does not have cross section data");
    } else {
      LOG_INFO("Material ", material.getName(), " has cross section data");
      material.validateXSec();
    }
  }

  // Create the pin meshes and coarse cells
  for (Int i = 0; i < mesh_types_ids.size(); ++i) {
    // Get the index of the i-th mesh
    Int idx = 0;
    for (auto const & cc_id : coarse_cell_ids) {
      if (cc_id == i) {
        break;
      }
      ++idx;
    }
    ASSERT(idx < coarse_cell_ids.size());
    auto const mesh_type = mesh_types_ids[idx].first;
    auto const mesh_id = mesh_types_ids[idx].second;
    Vec2F const & xy_extent = xy_extents[idx];
    model.addCoarseCell(xy_extent, mesh_type, mesh_id, coarse_cell_material_ids[idx]);
  }

  // Create the RTMs
  for (Int i = 0; i < rtm_ids.size(); ++i) {
    // Get the index of the i-th RTM
    Int idx = 0;
    for (auto const & rtm_id : rtm_ids) {
      if (rtm_id == i) {
        break;
      }
      ++idx;
    }
    ASSERT(idx < rtm_ids.size());
    model.addRTM(rtm_coarse_cell_ids[idx]);
  }

  // Create the lattices
  for (Int i = 0; i < lattice_ids.size(); ++i) {
    // Get the index of the i-th lattice
    Int idx = 0;
    for (auto const & lat_id : lattice_ids) {
      if (lat_id == i) {
        break;
      }
      ++idx;
    }
    ASSERT(idx < lattice_ids.size());
    model.addLattice(lattice_rtm_ids[idx]);
  }

  // Create the assemblies
  for (Int i = 0; i < assembly_ids.size(); ++i) {
    // Get the index of the i-th assembly
    Int idx = 0;
    for (auto const & asy_id : assembly_ids) {
      if (asy_id == i) {
        break;
      }
      ++idx;
    }
    ASSERT(idx < assembly_ids.size());
    model.addAssembly(assembly_lattice_ids[idx], assembly_lattice_zs[idx]);
  }

  // Create the core
  model.addCore(core_assembly_ids);
}

} // namespace

//==============================================================================
// read
//==============================================================================

void
Model::read(String const & filename)
{
  if (filename.ends_with(".xdmf")) {
    readXDMFFile(filename, *this);
  } else {
    logger::error("Unsupported file format.");
  }
}

//==============================================================================
// getCoarseCellOpticalThickness
//==============================================================================

// template <Int P, Int N>
// static void
// computeCellOpticalThickness(
//     FaceVertexMesh<P, N> const & fvm,
//     Vec2F const xy_extents,
//     Vector<MatID> const & material_ids,
//     Vector<Material> const & materials,
//     Vector<Float> & taus)
//{
//   // In a domain \Omega, with boundary \partial\Omega, the mean optical thickness
//   // in energy group g over the domain is defined as:
//   //    \tau_g = \frac{\pi}{P_{\partial\Omega}} * \sum_{i=1}^N A_i * \Sigma_{i,g}
//   //
//   // where:
//   //  P_{\partial\Omega} is the perimeter of the boundary of the domain
//   //  N is the number of faces of the mesh which fills \Omega
//   //  A_i is the area of the i-th face
//   //  \Sigma_{i,g} is the macroscopic cross section of the i-th face at energy group g
//
//
//   // Setup taus
//   Int const num_groups = materials[0].xsec().t().size();
//   Float constexpr zero = 0;
//   taus.resize(num_groups);
//   um2::fill(taus.begin(), taus.end(), zero);
//
//   Int const num_faces = fvm.numFaces();
//   for (Int iface = 0; iface < num_faces; ++iface) {
//     auto const a_i = fvm.getFace(iface).area();
//     auto const mat_id = material_ids[iface];
//     auto const & xsec = materials[static_cast<Int>(mat_id)].xsec();
//     for (Int ig = 0; ig < num_groups; ++ig) {
//       taus[ig] += a_i * xsec.t(ig);
//     }
//   }
//
//   Float const pi_over_p = um2::pi_2<Float> / (xy_extents[0] + xy_extents[1]);
//   for (Int ig = 0; ig < num_groups; ++ig) {
//     taus[ig] *= pi_over_p;
//   }
// }

// void
// Model::getCoarseCellOpticalThickness(Int const cc_id, Vector<Float> & taus) const
//{
//   // Check that the coarse cell exists
//   ASSERT(cc_id >= 0);
//   ASSERT(cc_id < _coarse_cells.size());
//
//   // Get the coarse cell
//   auto const & cc = _coarse_cells[cc_id];
//
//   // Call the appropriate function based on the mesh type
//   switch (cc.mesh_type) {
//     case MeshType::Tri:
//       {
//       auto const & tri_mesh = getTriMesh(cc.mesh_id);
//       computeCellOpticalThickness(tri_mesh, cc.xy_extents, cc.material_ids, _materials,
//       taus);
//       }
//       break;
//     case MeshType::Quad:
//       {
//       auto const & quad_mesh = getQuadMesh(cc.mesh_id);
//       computeCellOpticalThickness(quad_mesh, cc.xy_extents, cc.material_ids,
//       _materials, taus);
//       }
//       break;
//     case MeshType::QuadraticTri:
//       {
//       auto const & tri6_mesh = getTri6Mesh(cc.mesh_id);
//       computeCellOpticalThickness(tri6_mesh, cc.xy_extents, cc.material_ids,
//       _materials, taus);
//       }
//       break;
//     case MeshType::QuadraticQuad:
//       {
//       auto const & quad8_mesh = getQuad8Mesh(cc.mesh_id);
//       computeCellOpticalThickness(quad8_mesh, cc.xy_extents, cc.material_ids,
//       _materials, taus);
//       }
//       break;
//     default:
//       logger::error("Unsupported mesh type");
//   }
// }

//==============================================================================
// getCoarseCellHomogenizedXSec
//==============================================================================

namespace
{

template <Int P, Int N>
auto
computeCellHomogenizedXSec(FaceVertexMesh<P, N> const & fvm,
                           Vector<MatID> const & material_ids,
                           Vector<Material> const & materials) -> XSec
{
  Int const num_groups = materials[0].xsec().numGroups();
  Float constexpr zero = 0;
  Vector<Float> areas(materials.size(), zero);

  Int const num_faces = fvm.numFaces();
  for (Int iface = 0; iface < num_faces; ++iface) {
    auto const a_i = fvm.getFace(iface).area();
    auto const mat_id = material_ids[iface];
    areas[static_cast<Int>(mat_id)] += a_i;
  }

  XSec result(num_groups);
  // For each material with non-zero area, reduce into result
  for (Int imat = 0; imat < materials.size(); ++imat) {
    // Want exact comparison to zero
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    if (areas[imat] == zero) {
      continue;
    }
#pragma GCC diagnostic pop
    auto const & xsec = materials[imat].xsec();
    // a, f, nuf, tr, s, ss
    for (Int ig = 0; ig < num_groups; ++ig) {
      result.a()[ig] += areas[imat] * xsec.a()[ig];
      result.f()[ig] += areas[imat] * xsec.f()[ig];
      result.nuf()[ig] += areas[imat] * xsec.nuf()[ig];
      result.tr()[ig] += areas[imat] * xsec.tr()[ig];
      result.s()[ig] += areas[imat] * xsec.s()[ig];
      for (Int jg = 0; jg < num_groups; ++jg) {
        result.ss()(jg, ig) += areas[imat] * xsec.ss()(jg, ig);
      }
    }
  }

  // Normalize by the total area
  auto const total_area = um2::sum(areas.cbegin(), areas.cend());
  auto const inv_total_area = 1 / total_area;
  for (Int ig = 0; ig < num_groups; ++ig) {
    result.a()[ig] *= inv_total_area;
    result.f()[ig] *= inv_total_area;
    if (result.f()[ig] > 0) {
      result.isFissile() = true;
    }
    result.nuf()[ig] *= inv_total_area;
    result.tr()[ig] *= inv_total_area;
    result.s()[ig] *= inv_total_area;
    for (Int jg = 0; jg < num_groups; ++jg) {
      result.ss()(jg, ig) *= inv_total_area;
    }
  }

  result.isMacro() = true;
  result.validate();
  return result;
}

} // namespace

PURE auto
Model::getCoarseCellHomogenizedXSec(Int cc_id) const -> XSec
{
  LOG_INFO("Getting homogenized cross section for coarse cell ", cc_id);

  // For each face in the coarse cell, get the material ID and area. Keep track
  // of the area of each material in the cell, then at the end, use area
  // weighting to compute the homogenized cross section.
  ASSERT(cc_id >= 0);
  ASSERT(cc_id < _coarse_cells.size());

  // Get the coarse cell
  auto const & cc = _coarse_cells[cc_id];

  // Call the appropriate function based on the mesh type
  switch (cc.mesh_type) {
  case MeshType::Tri: {
    auto const & tri_mesh = getTriMesh(cc.mesh_id);
    return computeCellHomogenizedXSec(tri_mesh, cc.material_ids, _materials);
  }
  case MeshType::Quad: {
    auto const & quad_mesh = getQuadMesh(cc.mesh_id);
    return computeCellHomogenizedXSec(quad_mesh, cc.material_ids, _materials);
  }
  case MeshType::QuadraticTri: {
    auto const & tri6_mesh = getTri6Mesh(cc.mesh_id);
    return computeCellHomogenizedXSec(tri6_mesh, cc.material_ids, _materials);
  }
  case MeshType::QuadraticQuad: {
    auto const & quad8_mesh = getQuad8Mesh(cc.mesh_id);
    return computeCellHomogenizedXSec(quad8_mesh, cc.material_ids, _materials);
  }
  default:
    logger::error("Unsupported mesh type");
  }
  return {};
} // getCoarseCellHomogenizedXSec

PURE auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::getMeanChordLengths() const -> Vector<Float>
{
  Int total_num_cells = 0;
  for (auto const & asy_id : _core.children()) {
    for (auto const & lat_id : _assemblies[asy_id].children()) {
      for (auto const & rtm_id : _lattices[lat_id].children()) {
        for (auto const & cc_id : _rtms[rtm_id].children()) {
          total_num_cells += _coarse_cells[cc_id].numFaces();
        }
      }
    }
  }
  Vector<Float> result(total_num_cells, 0);
  Int ctr = 0;
  for (auto const & asy_id : _core.children()) {
    for (auto const & lat_id : _assemblies[asy_id].children()) {
      for (auto const & rtm_id : _lattices[lat_id].children()) {
        for (auto const & cc_id : _rtms[rtm_id].children()) {
          // Get the coarse cell
          auto const & cc = _coarse_cells[cc_id];

          // Call the appropriate function based on the mesh type
          switch (cc.mesh_type) {
          case MeshType::Tri: {
            auto const & mesh = getTriMesh(cc.mesh_id);
            Int const num_faces = mesh.numFaces();
            for (Int i = 0; i < num_faces; ++i) {
              result[ctr + i] = mesh.getFace(i).meanChordLength();
            }
            ctr += num_faces;
            break;
          }
          case MeshType::Quad: {
            auto const & mesh = getQuadMesh(cc.mesh_id);
            Int const num_faces = mesh.numFaces();
            for (Int i = 0; i < num_faces; ++i) {
              result[ctr + i] = mesh.getFace(i).meanChordLength();
            }
            ctr += num_faces;
            break;
          }
          case MeshType::QuadraticTri: {
            auto const & mesh = getTri6Mesh(cc.mesh_id);
            Int const num_faces = mesh.numFaces();
            for (Int i = 0; i < num_faces; ++i) {
              result[ctr + i] = mesh.getFace(i).meanChordLength();
            }
            ctr += num_faces;
            break;
          }
          case MeshType::QuadraticQuad: {
            auto const & mesh = getQuad8Mesh(cc.mesh_id);
            Int const num_faces = mesh.numFaces();
            for (Int i = 0; i < num_faces; ++i) {
              result[ctr + i] = mesh.getFace(i).meanChordLength();
            }
            ctr += num_faces;
            break;
          }
          default:
            logger::error("Unsupported mesh type");
          }
        }
      }
    }
  }
  return result;
} // getMeanChordLengths

//==============================================================================
// writeCMFDInfo
//==============================================================================

#if UM2_HAS_CMFD
void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::writeCMFDInfo(String const & filename) const
{
  // Use capital letters for matrices
  // NOLINTBEGIN(readability-identifier-naming)
  LOG_INFO("Writing MPACT model CMFD data to ", filename);

  PolytopeSoup soup;

  auto const & core = _core;
  auto const & assemblies = _assemblies;
  auto const & lattices = _lattices;
  auto const & rtms = _rtms;
  auto const & coarse_cells = _coarse_cells;
  auto const & materials = _materials;

  // Validate the XSecs of each material and ensure they have the same number of groups
  Int const num_groups = materials[0].xsec().numGroups();
  for (auto const & mat : materials) {
    mat.validateXSec();
    ASSERT(mat.xsec().numGroups() == num_groups);
  }

  if (core.children().empty()) {
    logger::error("Core has no children");
    return;
  }

  // Store:
  //  group-wise homogenized xsecs for each coarse cell
  //  //  group-wise optical thicknesses for each coarse cell
  //  //    (pi * area * homogenized xsec / perimeter)
  //  vector of face IDs for each coarse cell
  Int const num_coarse_cells = coarse_cells.size();
  Vector<XSec> coarse_cell_xsecs(num_coarse_cells);
  for (Int icc = 0; icc < num_coarse_cells; ++icc) {
    coarse_cell_xsecs[icc] = getCoarseCellHomogenizedXSec(icc);
  }

  Vector<Vector<Int>> coarse_cell_face_ids(num_coarse_cells);

  // Store the IDs of the vertices of each face
  Vector<Int> conn(4);

  // For each assembly
  Int const nyasy = core.grid().numCells(1);
  Int const nxasy = core.grid().numCells(0);
  for (Int iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Int ixasy = 0; ixasy < nxasy; ++ixasy) {

      // Get the assembly ID
      auto const asy_id = core.getChild(ixasy, iyasy);
      ASSERT(asy_id >= 0);
      ASSERT(asy_id < assemblies.size());

      // Get the assembly offset (lower left corner)
      Point2F const asy_ll = core.grid().getBox(ixasy, iyasy).minima();

      // Get the assembly
      auto const & assembly = assemblies[asy_id];
      if (assembly.children().empty()) {
        logger::error("Assembly has no children");
        return;
      }

      // For each lattice
      Int const nzlat = assembly.grid().numCells(0);
      for (Int izlat = 0; izlat < nzlat; ++izlat) {

        // Get the lattice ID
        auto const lat_id = assembly.getChild(izlat);
        ASSERT(lat_id >= 0);
        ASSERT(lat_id < lattices.size());

        // Get the lattice offset (z direction)
        // The midplane is the location that the geometry was sampled at.
        Float const low_z = assembly.grid().divs(0)[izlat];
        Float const high_z = assembly.grid().divs(0)[izlat + 1];
        Float const lat_z = (low_z + high_z) / 2;

        // Get the lattice
        auto const & lattice = lattices[lat_id];
        if (lattice.children().empty()) {
          logger::error("Lattice has no children");
          return;
        }

        // For each RTM
        Int const nyrtm = lattice.grid().numCells(1);
        Int const nxrtm = lattice.grid().numCells(0);
        for (Int iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (Int ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {

            // Get the RTM ID
            auto const rtm_id = lattice.getChild(ixrtm, iyrtm);
            ASSERT(rtm_id >= 0);
            ASSERT(rtm_id < rtms.size());

            // Get the RTM offset (lower left corner)
            Point2F const rtm_ll = lattice.grid().getBox(ixrtm, iyrtm).minima();

            // Get the rtm
            auto const & rtm = rtms[rtm_id];
            if (rtm.children().empty()) {
              logger::error("RTM has no children");
              return;
            }

            // For each coarse cell
            Int const nycells = rtm.grid().numCells(1);
            Int const nxcells = rtm.grid().numCells(0);
            for (Int iycell = 0; iycell < nycells; ++iycell) {
              for (Int ixcell = 0; ixcell < nxcells; ++ixcell) {

                // Get the coarse cell ID
                auto const & cell_id = rtm.getChild(ixcell, iycell);
                ASSERT(cell_id >= 0);
                ASSERT(cell_id < coarse_cells.size());

                // Get the cell box
                auto const cell_bb = rtm.grid().getBox(ixcell, iycell);
                // Get the cell offset (lower left corner)
                Point2F const cell_ll = cell_bb.minima();
                // Get the cell extents
                Vec2F const cell_extents = cell_bb.extents();

                // Compute the global xyz offset
                Point2F const xy_offset = cell_ll + rtm_ll + asy_ll;

                // Add the quad to the soup
                Point3F const v0(xy_offset[0], xy_offset[1], lat_z);
                Point3F const v1(xy_offset[0] + cell_extents[0], xy_offset[1], lat_z);
                Point3F const v2(xy_offset[0] + cell_extents[0],
                                 xy_offset[1] + cell_extents[1], lat_z);
                Point3F const v3(xy_offset[0], xy_offset[1] + cell_extents[1], lat_z);

                conn[0] = soup.addVertex(v0);
                conn[1] = soup.addVertex(v1);
                conn[2] = soup.addVertex(v2);
                conn[3] = soup.addVertex(v3);

                auto const face_id = soup.addElement(VTKElemType::Quad, conn);

                // add the face ID to coarse_cell_face_ids
                auto & cc_face_ids = coarse_cell_face_ids[cell_id];
                cc_face_ids.emplace_back(face_id);

              } // for (ixcell)
            } // for (iycell)
          } // for (ixrtm)
        } // for (iyrtm)
      } // for (izlat)
    } // for (ixasy)
  } // for (iyasy)

  // add elsets for each coarse cell
  for (Int icc = 0; icc < num_coarse_cells; ++icc) {
    String const elset_name = "Coarse_Cell_" + getASCIINumber(icc);
    auto const & cc_face_ids = coarse_cell_face_ids[icc];
    soup.addElset(elset_name, cc_face_ids);
  } // for (icc)

  auto const num_faces = soup.numElements();
  Vector<Int> ids(num_faces);
  um2::iota(ids.begin(), ids.end(), 0);
  Vector<Float> data(num_faces);

  // Construct the group g total cross section elsets
  for (Int ig = 0; ig < num_groups; ++ig) {
    um2::fill(data.begin(), data.end(), -infDistance<Float>());
    String const elset_name = "Homogenized_Total_XS_Group_" + getASCIINumber(ig);
    for (Int icc = 0; icc < num_coarse_cells; ++icc) {
      auto const sigma_t = coarse_cell_xsecs[icc].t(ig);
      auto const & face_ids = coarse_cell_face_ids[icc];
      for (auto const & face_id : face_ids) {
        data[face_id] = sigma_t;
      }
    }
    soup.addElset(elset_name, ids, data);
#  if UM2_ENABLE_ASSERTS
    // check that all data has been filled
    for (auto const & d : data) {
      ASSERT(d > -infDistance<Float>());
    }
#  endif
  }

  // Construct the group g scattering ratio elsets
  for (Int ig = 0; ig < num_groups; ++ig) {
    um2::fill(data.begin(), data.end(), -infDistance<Float>());
    String const elset_name = "Homogenized_Scattering_Ratio_Group_" + getASCIINumber(ig);
    for (Int icc = 0; icc < num_coarse_cells; ++icc) {
      auto const sigma_t = coarse_cell_xsecs[icc].t(ig);
      auto const sigma_s = coarse_cell_xsecs[icc].s()[ig];
      auto const c = sigma_s / sigma_t;
      auto const & face_ids = coarse_cell_face_ids[icc];
      for (auto const & face_id : face_ids) {
        data[face_id] = c;
      }
    }
    soup.addElset(elset_name, ids, data);
#  if UM2_ENABLE_ASSERTS
    // check that all data has been filled
    for (auto const & d : data) {
      ASSERT(d > -infDistance<Float>());
    }
#  endif
  }

  Int constexpr p = 4;
  Matrix<ComplexF> An(p, p);
  Matrix<ComplexF> Bn(p, p);
  An.zero();
  Bn.zero();
  Matrix<ComplexF> U(p, p);
  Matrix<ComplexF> omega(p, p);
  Matrix<ComplexF> J(p, p);
  um2::fill(J.begin(), J.end(), ComplexF(1.0, 0.0));
  Matrix<ComplexF> I(p, p); // Identity matrix
  Vector<Int> ipiv(p);

  // Construct the group g 1D CMFD spectral radius
  for (Int ig = 0; ig < num_groups; ++ig) {
    um2::fill(data.begin(), data.end(), -infDistance<Float>());
    String const elset_name = "1D_odCMFD_Spectral_Radius_Group_" + getASCIINumber(ig);
    for (Int icc = 0; icc < num_coarse_cells; ++icc) {
      // LOG_INFO("Computing spectral radius for coarse cell ", icc, " group ", ig);
      auto const sigma_t = coarse_cell_xsecs[icc].t(ig);
      auto const sigma_s = coarse_cell_xsecs[icc].s()[ig];
      auto const c = sigma_s / sigma_t;

      // CMFD Cell Parameters
      // Use the average side length of the cell as the width of the 1D cell
      auto const & cc = getCoarseCell(icc);
      auto const & xy_extents = cc.xy_extents;
      auto const w = (xy_extents[0] + xy_extents[1]) / 2;
      // Fix number of cells at 4
      Int const s = 1;        // number of sweeps
      Float const eta = 0.25; // odcmfd

      Complex<Float> rho(-1, 0);
      if (c > 1) {
        LOG_WARN("Scattering ratio is greater than 1 for coarse cell ", icc, " group ",
                 ig, ". Setting rho to -1.");
      } else {
        CMFDCellParams const params(w, p, sigma_t, c, s, eta);
        //        rho = spectral_radius(params);
        rho = spectral_radius(params, An, Bn, U, omega, J, I, ipiv);
      }
      auto const & face_ids = coarse_cell_face_ids[icc];
      for (auto const & face_id : face_ids) {
        data[face_id] = rho.real();
      }
    }
    soup.addElset(elset_name, ids, data);
#  if UM2_ENABLE_ASSERTS
    // check that all data has been filled
    for (auto const & d : data) {
      ASSERT(d > -infDistance<Float>());
    }
#  endif
  }

  soup.write(filename);
  // NOLINTEND(readability-identifier-naming)
} // writeCMFDInfo

#endif // UM2_HAS_CMFD

} // namespace um2::mpact

// NOLINTEND(misc-include-cleaner)
