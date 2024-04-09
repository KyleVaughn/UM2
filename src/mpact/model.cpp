#include <um2/mpact/model.hpp>

#include <um2/common/logger.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>
#include <um2/stdlib/algorithm/fill.hpp>
#include <um2/stdlib/numeric/iota.hpp>

#include <algorithm> // std::any_of
#include <numeric> // std::reduce

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
template <typename T, typename U>
static void
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
Model::addMaterial(Material const & material) -> Int
{
  material.validate();
  _materials.emplace_back(material);
  return _materials.size() - 1;
}

//=============================================================================
// addCylindricalPinMesh
//=============================================================================

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::addCylindricalPinMesh(
    Float const pitch,
    Vector<Float> const & radii,
    Vector<Int> const & num_rings,
    Int const num_azimuthal,
    Int const mesh_order) -> Int
{

  Int mesh_id = -1;
  if (mesh_order == 1) {
    mesh_id = this->_quads.size();
    logger::info("Adding quad cylindrical pin mesh ", mesh_id);
  } else if (mesh_order == 2) {
    mesh_id = this->_quad8s.size();
    logger::info("Adding quad8 cylindrical pin mesh ", mesh_id);
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

  if (std::any_of(radii.begin(), radii.end(), [pitch](Float r) { return r > pitch / 2; })) {
    logger::error("The radii must be less than half the pitch");
    return -1;
  }

  Float constexpr eps = eps_distance;
  Float constexpr big_eps = 100 * eps;

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
    radial_region_areas[i] = pi<Float> * (radii[i] * radii[i] - radii[i - 1] * radii[i - 1]);
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
      ring_radii[ctr] =
          um2::sqrt(area_per_ring / pi<Float> + ring_radii[ctr - 1] * ring_radii[ctr - 1]);
    }
  }
  // Outside of the last ring
  ring_areas[ctr] = pitch * pitch - pi<Float> * ring_radii.back() * ring_radii.back();
  // Log the radii and areas in debug mode
  for (Int i = 0; i < total_rings; ++i) {
    LOG_DEBUG("Ring ", i ," radius: ", ring_radii[i]);
    LOG_DEBUG("Ring ", i ," area: ", ring_areas[i]);
  }
  LOG_DEBUG("The area outside of the last ring is ", ring_areas[ctr]);
  // Ensure the sum of the ring areas is equal to pitch^2
  Float const sum_ring_areas =
      std::reduce(ring_areas.begin(), ring_areas.end());
  ASSERT_NEAR(sum_ring_areas, pitch * pitch, eps);
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
    ASSERT_NEAR(eq_radii[0] * eq_radii[0] * sin_theta / 2,
                ring_areas[0] / num_azimuthal_t, big_eps);
    for (Int i = 1; i < total_rings; ++i) {
      Float const area =
          (eq_radii[i] * eq_radii[i] - eq_radii[i - 1] * eq_radii[i - 1]) * sin_theta / 2;
      ASSERT_NEAR(area, ring_areas[i] / num_azimuthal_t, big_eps);
    }

    //------------------------------------------------------------------------
    // Get the points that make up the mesh
    //------------------------------------------------------------------------
    // The number of points is:
    //   Center point
    //   num_azimuthal / 2, for the points in the innermost ring to make the quads
    //      "triangular"
    //   (num_rings + 1) * num_azimuthal
    Int const num_points = 1 + (total_rings + 1) * num_azimuthal + num_azimuthal / 2;
    Vector<Point2> vertices(num_points);
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
      Float sin_ia_theta = std::sin(theta * static_cast<Float>(ia));
      Float cos_ia_theta = std::cos(theta * static_cast<Float>(ia));
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
      logger::debug("Ring ", i, " equivalent radius: ", eq_radii[i]);
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
    Vector<Point2> vertices(num_points);
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
      logger::debug("Point ", i, ": ", vertices[i][0], ", ", vertices[i][1]);
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
  logger::error("Only linear and quadratic meshes are supported for a cylindrical pin mesh");
  return -1;
}

//=============================================================================
// addRectangularPinMesh
//=============================================================================

auto
Model::addRectangularPinMesh(Vec2F const xy_extents, Int const nx_faces, Int const ny_faces) -> Int
{
  Int const mesh_id = _quads.size();
  logger::info("Adding rectangular pin mesh ", mesh_id);
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
// addCoarseCell
//=============================================================================

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::addCoarseCell(Vec2F const xy_extents,
    MeshType const mesh_type,
    Int const mesh_id,
    Vector<MatID> const & material_ids) -> Int
{
  Int const cc_id = _coarse_cells.size();
  logger::info("Adding coarse cell ", cc_id);
  // Ensure extents are positive
  if (xy_extents[0] <= 0 || xy_extents[1] <= 0) {
    logger::error("Coarse cell dimensions must be positive");
    return -1;
  }
  // Ensure that the mesh exists and the material IDs are valid
  if (mesh_id != -1) {
    switch (mesh_type) {
    case MeshType::Tri:
      if (0 < mesh_id || mesh_id >= _tris.size()) {
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
  logger::info("Adding ray tracing module ", rtm_id);
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
  RectilinearGrid2 const grid(xy_extents, cc_ids_renumbered);
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
  logger::info("Adding lattice ", lat_id);
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
  Point2 const minima(0, 0);
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
  RegularGrid2 const grid(minima, spacing, num_cells);
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
  logger::info("Adding assembly ", asy_id);
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
  if (um2::abs(z.front() + ahalf) > eps_distance
      || um2::abs(z.back() - ahalf) > eps_distance) {
    if (um2::abs(z.front()) > eps_distance) {
      logger::error("The first z-plane must be at 0");
      return -1;
    }
  }
  // Ensure this assembly is the same height as all other assemblies
  if (!_assemblies.empty()) {
    auto const assem_top = _assemblies[0].grid().maxima(0);
    auto const assem_bot = _assemblies[0].grid().minima(0);
    if (um2::abs(z.back() - assem_top) > eps_distance ||
        um2::abs(z.front() - assem_bot) > eps_distance) {
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

  RectilinearGrid1 grid;
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
  logger::info("Adding core");
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
  RectilinearGrid2 const grid(xy_extents, asy_ids);
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
    AxisAlignedBox2 bb = AxisAlignedBox2::empty();
    Point2 * vertices = nullptr;
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
    }

    // Shift the points so that the min point is at the origin.
    Point2 const min_point = bb.minima();
    for (Int ip = 0; ip < num_verts; ++ip) {
      vertices[ip] -= min_point;
    }
#if UM2_ENABLE_ASSERTS
    Point2 const dxdy = bb.maxima() - bb.minima();
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
    auto const xs = _materials[imat].xsec().collapse();
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
      String const asy_name = "Assembly_" + getASCIINumber(asy_id) + "_" + getASCIINumber(asy_id_ctr);
      LOG_DEBUG("Assembly name: ", asy_name);

      // Get the assembly offset (lower left corner)
      Point2 const asy_ll = _core.grid().getBox(ixasy, iyasy).minima();

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
        String const lat_name = "Lattice_" + getASCIINumber(lat_id) + "_" + getASCIINumber(lat_id_ctr);
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
            String const rtm_name = "RTM_" + getASCIINumber(rtm_id) + "_" + getASCIINumber(rtm_id_ctr);
            LOG_DEBUG("RTM name: ", rtm_name);

            // Get the RTM offset (lower left corner)
            Point2 const rtm_ll = lattice.grid().getBox(ixrtm, iyrtm).minima();

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
                String const cell_name = "Coarse_Cell_" + getASCIINumber(cell_id) + "_" + getASCIINumber(cell_id_ctr);
                LOG_DEBUG("Coarse cell name: ", cell_name);

                // Get the cell offset (lower left corner)
                Point2 const cell_ll = rtm.grid().getBox(ixcell, iycell).minima();

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
                Point2 const xy_offset = cell_ll + rtm_ll + asy_ll;
                Point3 const global_offset = Point3(xy_offset[0], xy_offset[1], lat_z);
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
            }   // for (iycell)
            lattice_soup += rtm_soup;
          }   // for (ixrtm)
        }     // for (iyrtm)
        assembly_soup += lattice_soup;
      } // for (izlat)
      core_soup += assembly_soup;
    } // for (ixasy)
  }   // for (iyasy)
  if (core_soup.numElsets() != 0) {
    core_soup.sortElsets();
  }
  return core_soup;
} // operator PolytopeSoup

//==============================================================================
// writeXDMF
//==============================================================================

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
Model::writeXDMFFile(String const & filepath) const
{
  LOG_INFO("Writing MPACT model to XDMF file: ", filepath);

  if (_core.children().empty()) {
    logger::error("Core has no children");
    return;
  }

  // Store the one-group cross sections for each material
  Vector<Float> one_group_xs(_materials.size());
  for (Int imat = 0; imat < _materials.size(); ++imat) {
    ASSERT(_materials[imat].xsec().isMacro());
    auto const xs = _materials[imat].xsec().collapse();
    one_group_xs[imat] = xs.t(0);
  }

  // Store a PolytopeSoup for each CoarseCell
  Vector<PolytopeSoup> coarse_cell_soups(_coarse_cells.size());
  {
    Vector<Float> mcls;
    Vector<Int> cell_ids;
    Vector<Float> mat_ids;
    Vector<Float> xsecs;
    for (Int icc = 0; icc < _coarse_cells.size(); ++icc) {
      auto & cell_soup = coarse_cell_soups[icc];
      auto const & coarse_cell = _coarse_cells[icc];

      // Get the mesh type and id of the coarse cell.
      MeshType const mesh_type = coarse_cell.mesh_type;
      Int const mesh_id = coarse_cell.mesh_id;

      mcls.resize(coarse_cell.numFaces());

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

      // Add the one-group cross sections
      xsecs.resize(coarse_cell.numFaces());
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
  for (auto const & material : _materials) {
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
  Vector<Int> asy_found(_assemblies.size(), -1);
  Vector<Int> lat_found(_lattices.size(), -1);
  Vector<Int> rtm_found(_rtms.size(), -1);
  Vector<Int> cc_found(_coarse_cells.size(), -1);

  Int const nyasy = _core.grid().numCells(1);
  Int const nxasy = _core.grid().numCells(0);

  // Core NX_by_NY
  pugi::xml_node xcore_info = xcore_grid.append_child("Information");
  xcore_info.append_attribute("Name") = "NX_by_NY";
  String const core_nx_by_ny = String(nxasy) + " x " + String(nyasy); 
  xcore_info.append_child(pugi::node_pcdata).set_value(core_nx_by_ny.data());

  // For each assembly
  for (Int iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Int ixasy = 0; ixasy < nxasy; ++ixasy) {

      // Get the assembly ID
      auto const asy_id = _core.getChild(ixasy, iyasy);
      ASSERT(asy_id >= 0);
      ASSERT(asy_id < _assemblies.size());

      // Increment the number of times we have seen this assembly
      Int const asy_id_ctr = ++asy_found[asy_id];

      // Get the assembly name
      String const asy_name = "Assembly_" + getASCIINumber(asy_id) + "_" + getASCIINumber(asy_id_ctr);
      LOG_DEBUG("Assembly name: ", asy_name);

      // Create the assembly group
      pugi::xml_node xasy_grid = xcore_grid.append_child("Grid");
      xasy_grid.append_attribute("Name") = asy_name.data();
      xasy_grid.append_attribute("GridType") = "Tree";
      String const h5asy_grouppath = h5core_grouppath + "/" + asy_name;
      H5::Group const h5asy_group = h5file.createGroup(h5asy_grouppath.data());

      // Get the assembly offset (lower left corner)
      Point2 const asy_ll = _core.grid().getBox(ixasy, iyasy).minima();

      // Get the assembly
      auto const & assembly = _assemblies[asy_id];
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
        ASSERT(lat_id < _lattices.size());

        // Increment the number of times we have seen this lattice
        Int const lat_id_ctr = ++lat_found[lat_id];

        // Get the lattice name
        String const lat_name = "Lattice_" + getASCIINumber(lat_id) + "_" + getASCIINumber(lat_id_ctr);
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
        String const lat_z_str = String(low_z) + ", " + String(lat_z) + ", " + String(high_z);
        xlat_info.append_child(pugi::node_pcdata).set_value(lat_z_str.data());

        // Get the lattice
        auto const & lattice = _lattices[lat_id];
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

        // For each RTM
        for (Int iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (Int ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {

            // Get the RTM ID
            auto const rtm_id = lattice.getChild(ixrtm, iyrtm);
            ASSERT(rtm_id >= 0);
            ASSERT(rtm_id < _rtms.size());

            // Increment the number of times we have seen this RTM
            Int const rtm_id_ctr = ++rtm_found[rtm_id];

            // Get the RTM name
            String const rtm_name = "RTM_" + getASCIINumber(rtm_id) + "_" + getASCIINumber(rtm_id_ctr);
            LOG_DEBUG("RTM name: ", rtm_name);

            // Create the RTM group
            pugi::xml_node xrtm_grid = xlat_grid.append_child("Grid");
            xrtm_grid.append_attribute("Name") = rtm_name.data();
            xrtm_grid.append_attribute("GridType") = "Tree";
            String const h5rtm_grouppath = h5lat_grouppath + "/" + rtm_name;
            H5::Group const h5rtm_group = h5file.createGroup(h5rtm_grouppath.data());

            // Get the RTM offset (lower left corner)
            Point2 const rtm_ll = lattice.grid().getBox(ixrtm, iyrtm).minima();

            // Get the rtm
            auto const & rtm = _rtms[rtm_id];
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

            // For each coarse cell
            for (Int iycell = 0; iycell < nycells; ++iycell) {
              for (Int ixcell = 0; ixcell < nxcells; ++ixcell) {

                // Get the coarse cell ID
                auto const & cell_id = rtm.getChild(ixcell, iycell);
                ASSERT(cell_id >= 0);
                ASSERT(cell_id < _coarse_cells.size());

                // Increment the number of times we have seen this coarse cell
                Int const cell_id_ctr = ++cc_found[cell_id];

                // Get the coarse cell name
                String const cell_name = "Coarse_Cell_" + getASCIINumber(cell_id) + "_" + getASCIINumber(cell_id_ctr);
                LOG_DEBUG("Coarse cell name: ", cell_name);

                // Get the cell offset (lower left corner)
                Point2 const cell_ll = rtm.grid().getBox(ixcell, iycell).minima();

                // Translate the cell_soup to the correct location
                // global xyz offset
                Point2 const xy_offset = cell_ll + rtm_ll + asy_ll;
                Point3 const global_offset = Point3(xy_offset[0], xy_offset[1], lat_z);

                auto const & cell_soup = coarse_cell_soups[cell_id];

                writeXDMFUniformGrid(cell_name, xrtm_grid, h5file, h5filename,
                    h5rtm_grouppath, cell_soup, global_offset);

              } // for (ixcell)
            }   // for (iycell)
          }   // for (ixrtm)
        }     // for (iyrtm)
      } // for (izlat)
    } // for (ixasy)
  }   // for (iyasy)

  // Write the XML file
  xdoc.save_file(filepath.data(), "  ");

  // Close the HDF5 file
  h5file.close();

  LOG_INFO("XDMF file written successfully");
} // writeXDMFFile

//==============================================================================
// write
//==============================================================================

void
Model::write(String const & filename) const
{
  if (filename.ends_with(".xdmf")) {
    writeXDMFFile(filename);
  } else {
    logger::error("Unsupported file format.");
  }
}

} // namespace um2::mpact
