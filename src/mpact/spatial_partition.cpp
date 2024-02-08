#include <um2/mpact/spatial_partition.hpp>

#include <iomanip> // std::setw
#include <numeric> // std::reduce

namespace um2::mpact
{

//=============================================================================
// checkMeshExists
//=============================================================================

void
SpatialPartition::checkMeshExists(MeshType mesh_type, I mesh_id) const
{
  switch (mesh_type) {
  case MeshType::Tri:
    if (0 > mesh_id || mesh_id >= this->_tris.size()) {
      log::error("Tri mesh ", mesh_id, " does not exist");
    }
    break;
  case MeshType::Quad:
    if (0 > mesh_id || mesh_id >= this->_quads.size()) {
      log::error("Quad mesh ", mesh_id, " does not exist");
    }
    break;
  case MeshType::QuadraticTri:
    if (0 > mesh_id || mesh_id >= this->_tri6s.size()) {
      log::error("Quadratic tri mesh ", mesh_id, " does not exist");
    }
    break;
  case MeshType::QuadraticQuad:
    if (0 > mesh_id || mesh_id >= this->_quad8s.size()) {
      log::error("Quadratic quad mesh ", mesh_id, " does not exist");
    }
    break;
  default:
    log::error("Invalid mesh type");
  }
}

#if UM2_ENABLE_ASSERTS
#  define CHECK_MESH_EXISTS(mesh_type, mesh_id) checkMeshExists(mesh_type, mesh_id)
#else
#  define CHECK_MESH_EXISTS(mesh_type, mesh_id)
#endif

//=============================================================================
// Accessors
//=============================================================================

PURE [[nodiscard]] auto
SpatialPartition::getTriMesh(I mesh_id) const noexcept -> TriFVM const &
{
  CHECK_MESH_EXISTS(MeshType::Tri, mesh_id);
  return _tris[mesh_id];
}

PURE [[nodiscard]] auto
SpatialPartition::getQuadMesh(I mesh_id) const noexcept -> QuadFVM const &
{
  CHECK_MESH_EXISTS(MeshType::Quad, mesh_id);
  return _quads[mesh_id];
}

PURE [[nodiscard]] auto
SpatialPartition::getTri6Mesh(I mesh_id) const noexcept -> Tri6FVM const &
{
  CHECK_MESH_EXISTS(MeshType::QuadraticTri, mesh_id);
  return _tri6s[mesh_id];
}

PURE [[nodiscard]] auto
SpatialPartition::getQuad8Mesh(I mesh_id) const noexcept -> Quad8FVM const &
{
  CHECK_MESH_EXISTS(MeshType::QuadraticQuad, mesh_id);
  return _quad8s[mesh_id];
}

//=============================================================================
// clear
//=============================================================================

HOSTDEV void
SpatialPartition::clear() noexcept
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
SpatialPartition::addMaterial(Material const & material) -> I
{
  _materials.push_back(material);
  return _materials.size() - 1;
}

//=============================================================================
// addCylindricalPinMesh
//=============================================================================

auto
// NOLINTNEXTLINE
SpatialPartition::addCylindricalPinMesh(Vector<F> const & radii, F const pitch,
                                         Vector<I> const & num_rings,
                                         I const num_azimuthal, I const mesh_order)
    -> I
{
  LOG_INFO("Making cylindrical pin mesh");
  if ((num_azimuthal & (num_azimuthal - 1)) != 0) {
    log::error("The number of azimuthal divisions must be a power of 2");
    return -1;
  }
  if (num_azimuthal < 8) {
    log::error("The number of azimuthal divisions must be at least 8");
    return -1;
  }
  if (radii.size() != num_rings.size()) {
    log::error("The number of radii must match the size of num_rings");
    return -1;
  }
  if (std::any_of(radii.begin(), radii.end(), [pitch](F r) { return r > pitch / 2; })) {
    log::error("The radii must be less than half the pitch");
    return -1;
  }

  I mesh_id = -1;
  if (mesh_order == 1) {
    mesh_id = this->_quads.size();
    log::info("Making linear quadrilateral cylindrical pin mesh ", mesh_id);
  } else if (mesh_order == 2) {
    mesh_id = this->_quad8s.size();
    log::info("Making quadratic quadrilateral cylindrical pin mesh ", mesh_id);
  } else {
    log::error("Invalid mesh order");
    return -1;
  }

  F constexpr eps = condCast<F>(1e-6); 
  F constexpr big_eps = condCast<F>(1e-4);

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
  //   ring_areas = the area of each ring, including the outside of the last radius

  //---------------------------------------------------------------------------
  // Get the area of each radial region (rings + outside of the last ring)
  //---------------------------------------------------------------------------
  I const num_radial_regions = radii.size() + 1;
  Vector<F> radial_region_areas(num_radial_regions);
  // A0 = pi * r0^2
  // Ai = pi * (ri^2 - ri-1^2)
  radial_region_areas[0] = pi<F> * radii[0] * radii[0];
  for (I i = 1; i < num_radial_regions - 1; ++i) {
    radial_region_areas[i] = pi<F> * (radii[i] * radii[i] - radii[i - 1] * radii[i - 1]);
  }
  radial_region_areas[num_radial_regions - 1] =
      pitch * pitch - radial_region_areas[num_radial_regions - 2];

  //---------------------------------------------------------------------------
  // Get the radii and areas of each ring after splitting the radial regions
  // This includes outside of the last ring
  //---------------------------------------------------------------------------
  I const total_rings = std::reduce(num_rings.begin(), num_rings.end(), 0);
  Vector<F> ring_radii(total_rings);
  Vector<F> ring_areas(total_rings + 1);
  // Inside the innermost region
  ring_areas[0] = radial_region_areas[0] / static_cast<F>(num_rings[0]);
  ring_radii[0] = um2::sqrt(ring_areas[0] / pi<F>);
  for (I i = 1; i < num_rings[0]; ++i) {
    ring_areas[i] = ring_areas[0];
    ring_radii[i] =
        um2::sqrt(ring_areas[i] / pi<F> + ring_radii[i - 1] * ring_radii[i - 1]);
  }
  I ctr = num_rings[0];
  for (I ireg = 1; ireg < num_radial_regions - 1; ++ireg) {
    I const num_rings_in_region = num_rings[ireg];
    F const area_per_ring =
        radial_region_areas[ireg] / static_cast<F>(num_rings_in_region);
    for (I iring = 0; iring < num_rings_in_region; ++iring, ++ctr) {
      ring_areas[ctr] = area_per_ring;
      ring_radii[ctr] =
          um2::sqrt(area_per_ring / pi<F> + ring_radii[ctr - 1] * ring_radii[ctr - 1]);
    }
  }
  // Outside of the last ring
  ring_areas[ctr] = pitch * pitch - pi<F> * ring_radii.back() * ring_radii.back();
  // Log the radii and areas in debug mode
  for (I i = 0; i < total_rings; ++i) {
    LOG_DEBUG("Ring ", i ," radius: ", ring_radii[i]);
    LOG_DEBUG("Ring ", i ," area: ", ring_areas[i]); 
  }
  LOG_DEBUG("The area outside of the last ring is ", ring_areas[ctr]);
  // Ensure the sum of the ring areas is equal to pitch^2
  F const sum_ring_areas =
      std::reduce(ring_areas.begin(), ring_areas.end());
  ASSERT_NEAR(sum_ring_areas, pitch * pitch, eps);
  F const num_azimuthal_t = static_cast<F>(num_azimuthal);
  if (mesh_order == 1) {
    // Get the equivalent radius of each ring if it were a quadrilateral
    F const theta = 2 * pi<F> / num_azimuthal_t;
    F const sin_theta = um2::sin(theta);
    Vector<F> eq_radii(total_rings);
    // The innermost radius is a special case, and is essentially a triangle.
    // A_t = l² * sin(θ) / 2
    // A_ring = num_azi * A_t = l² * sin(θ) * num_azi / 2
    // l = sqrt(2 * A_ring / (sin(θ) * num_azi))
    eq_radii[0] = um2::sqrt(2 * ring_areas[0] / (sin_theta * num_azimuthal_t));
    // A_q = (l² - l²₀) * sin(θ) / 2
    // A_ring = num_azi * A_q = (l² - l²₀) * sin(θ) * num_azi / 2
    // l = sqrt(2 * A_ring / (sin(θ) * num_azi) + l²₀)
    for (I i = 1; i < total_rings; ++i) {
      eq_radii[i] = um2::sqrt(2 * ring_areas[i] / (sin_theta * num_azimuthal_t) +
                              eq_radii[i - 1] * eq_radii[i - 1]);
    }
    for (I i = 0; i < total_rings; ++i) {
      LOG_DEBUG("Ring ", i, " equivalent radius: ", eq_radii[i]);
    }
    // If any of the equivalent radii are larger than half the pitch, error
    if (std::any_of(eq_radii.begin(), eq_radii.end(),
                    [pitch](F r) { return r > pitch / 2; })) {
      log::error("The equivalent radius of a ring is larger than half the pitch");
      return -1;
    }
    // Sanity check: ensure the sum of the quadrilateral areas in a ring is equal to
    // the ring area
    ASSERT_NEAR(eq_radii[0] * eq_radii[0] * sin_theta / 2,
                ring_areas[0] / num_azimuthal_t, big_eps);
    for (I i = 1; i < total_rings; ++i) {
      F const area =
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
    I const num_points = 1 + (total_rings + 1) * num_azimuthal + num_azimuthal / 2;
    Vector<Point2> vertices(num_points);
    LOG_DEBUG("The number of points is ", num_points);
    // Center point
    vertices[0] = {0, 0};
    // Triangular points
    LOG_DEBUG("Computing the triangular points");
    F const rt = eq_radii[0] / 2;
    for (I ia = 0; ia < num_azimuthal / 2; ++ia) {
      F const sin_ia_theta = um2::sin(theta * (2 * static_cast<F>(ia) + 1));
      F const cos_ia_theta = um2::cos(theta * (2 * static_cast<F>(ia) + 1));
      vertices[1 + ia] = {rt * cos_ia_theta, rt * sin_ia_theta};
    }
    LOG_DEBUG("Computing the quadrilateral points");
    // Quadrilateral points
    // Points on rings, not including the boundary of the pin (pitch / 2 box)
    for (I ir = 0; ir < total_rings; ++ir) {
      I const num_prev_points = 1 + num_azimuthal / 2 + ir * num_azimuthal;
      for (I ia = 0; ia < num_azimuthal; ++ia) {
        F sin_ia_theta = um2::sin(theta * static_cast<F>(ia));
        F cos_ia_theta = um2::cos(theta * static_cast<F>(ia));
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
    I const num_prev_points = 1 + num_azimuthal / 2 + total_rings * num_azimuthal;
    for (I ia = 0; ia < num_azimuthal; ++ia) {
      F sin_ia_theta = std::sin(theta * static_cast<F>(ia));
      F cos_ia_theta = std::cos(theta * static_cast<F>(ia));
      if (um2::abs(sin_ia_theta) < eps) {
        sin_ia_theta = 0;
      }
      if (um2::abs(cos_ia_theta) < eps) {
        cos_ia_theta = 0;
      }
      F const rx = um2::abs(pitch / (2 * cos_ia_theta));
      F const ry = um2::abs(pitch / (2 * sin_ia_theta));
      F const rb = um2::min(rx, ry);
      vertices[num_prev_points + ia] = {rb * cos_ia_theta, rb * sin_ia_theta};
    }

    //------------------------------------------------------------------------
    // Get the faces that make up the mesh
    //------------------------------------------------------------------------
    I const num_faces = num_azimuthal * (total_rings + 1);
    Vector<Vec<4, I>> faces(num_faces);
    // Establish a few aliases
    I const na = num_azimuthal;
    I const nr = total_rings;
    I const ntric = 1 + na / 2; // Number of triangular points + center point
    // Triangular quads
    for (I ia = 0; ia < na / 2; ++ia) {
      I const p0 = 0;                  // Center point
      I const p1 = ntric + ia * 2;     // Bottom right point on ring
      I const p2 = ntric + ia * 2 + 1; // Top right point on ring
      I const p3 = 1 + ia;             // The triangular point
      I p4 = ntric + ia * 2 + 2;       // Top left point on ring
      // If we're at the end of the ring, wrap around
      if (p4 == ntric + na) {
        p4 = ntric;
      }
      faces[2 * ia] = {p0, p1, p2, p3};
      faces[2 * ia + 1] = {p0, p3, p2, p4};
    }
    // Non-boundary and boundary quads
    for (I ir = 1; ir < nr + 1; ++ir) {
      for (I ia = 0; ia < na; ++ia) {
        I const p0 = ntric + (ir - 1) * na + ia; // Bottom left point
        I const p1 = ntric + (ir)*na + ia;       // Bottom right point
        I p2 = ntric + (ir)*na + ia + 1;         // Top right point
        I p3 = ntric + (ir - 1) * na + ia + 1;   // Top left point
        // If we're at the end of the ring, wrap around
        if (ia + 1 == na) {
          p2 -= na;
          p3 -= na;
        }
        faces[ir * na + ia] = {p0, p1, p2, p3};
      }
    }
    // Shift such that the lower left corner is at the origin
    F const half_pitch = pitch / 2;
    for (I i = 0; i < num_points; ++i) {
      vertices[i] += half_pitch;
      // Fix close to zero values
      if (um2::abs(vertices[i][0]) < eps) {
        vertices[i][0] = 0;
      }
      if (um2::abs(vertices[i][1]) < eps) {
        vertices[i][1] = 0;
      }
    }
    QuadFVM mesh(vertices, faces);
    //     mesh.populateVF();
    this->_quads.push_back(um2::move(mesh));
    LOG_DEBUG("Finished creating mesh");
    return mesh_id;
  }
  if (mesh_order == 2) {
    // Get the equivalent radius of each ring if it were a quadratic quadrilateral
    F const theta = 2 * pi<F> / num_azimuthal_t;
    F const gamma = theta / 2;
    F const sin_gamma = um2::sin(gamma);
    F const cos_gamma = um2::cos(gamma);
    F const sincos_gamma = sin_gamma * cos_gamma;
    Vector<F> eq_radii(total_rings);
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
    F const tri_area = ring_radii[0] * ring_radii[0] * sincos_gamma;
    F const ring_area = ring_areas[0] / num_azimuthal_t;
    F const l0 = ring_radii[0];
    F constexpr three_fourths = static_cast<F>(3) / static_cast<F>(4);
    eq_radii[0] =
        three_fourths * (ring_area - tri_area) / (l0 * sin_gamma) + l0 * cos_gamma;
    for (I i = 1; i < total_rings; ++i) {
      F const l_im1 = ring_radii[i - 1];
      F const ll_im1 = eq_radii[i - 1];
      F const a_edge_im1 =
          l_im1 * sin_gamma * (ll_im1 - l_im1 * cos_gamma) / three_fourths;
      F const l = ring_radii[i];
      F const a_quad = (l * l - l_im1 * l_im1) * sincos_gamma;
      F const a_ring = ring_areas[i] / num_azimuthal_t;
      eq_radii[i] = three_fourths * (a_ring - a_quad + a_edge_im1) / (l * sin_gamma) +
                    l * cos_gamma;
    }
    // Log the equivalent radii in debug mode
    for (I i = 0; i < total_rings; ++i) {
      log::debug("Ring ", i, " equivalent radius: ", eq_radii[i]);
    }
    // If any of the equivalent radii are larger than half the pitch, error
    if (std::any_of(eq_radii.begin(), eq_radii.end(),
                    [pitch](F r) { return r > pitch / 2; })) {
      log::error("The equivalent radius of a ring is larger than half the pitch.");
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
    I const na = num_azimuthal;
    I const nr = total_rings;
    I const num_points = 1 + 4 * na + 3 * na * nr;
    Vector<Point2> vertices(num_points);
    // Center point
    vertices[0] = {0, 0};
    // Triangular points
    F const rt = ring_radii[0] / 2;
    for (I ia = 0; ia < na; ++ia) {
      F const sin_ia_theta = um2::sin(static_cast<F>(ia) * theta);
      F const cos_ia_theta = um2::cos(static_cast<F>(ia) * theta);
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
    I num_prev_points = 1 + 2 * na;
    for (I ia = 0; ia < 2 * na; ++ia) {
      F sin_ia_gamma = um2::sin(static_cast<F>(ia) * gamma);
      F cos_ia_gamma = um2::cos(static_cast<F>(ia) * gamma);
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
    for (I ir = 1; ir < total_rings; ++ir) {
      num_prev_points = 1 + 4 * na + 3 * na * (ir - 1);
      // Between the rings
      for (I ia = 0; ia < num_azimuthal; ++ia) {
        F sin_ia_theta = um2::sin(static_cast<F>(ia) * theta);
        F cos_ia_theta = um2::cos(static_cast<F>(ia) * theta);
        if (um2::abs(sin_ia_theta) < eps) {
          sin_ia_theta = 0;
        }
        if (um2::abs(cos_ia_theta) < eps) {
          cos_ia_theta = 0;
        }
        F const r = (ring_radii[ir] + ring_radii[ir - 1]) / 2;
        vertices[num_prev_points + ia] = {r * cos_ia_theta, r * sin_ia_theta};
      }
      num_prev_points += num_azimuthal;
      for (I ia = 0; ia < 2 * num_azimuthal; ++ia) {
        F sin_ia_gamma = um2::sin(static_cast<F>(ia) * gamma);
        F cos_ia_gamma = um2::cos(static_cast<F>(ia) * gamma);
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
    for (I ia = 0; ia < num_azimuthal; ++ia) {
      F sin_ia_theta = um2::sin(static_cast<F>(ia) * theta);
      F cos_ia_theta = um2::cos(static_cast<F>(ia) * theta);
      if (um2::abs(sin_ia_theta) < eps) {
        sin_ia_theta = 0;
      }
      if (um2::abs(cos_ia_theta) < eps) {
        cos_ia_theta = 0;
      }
      // pitch and last ring radius
      F const rx = um2::abs(pitch / (2 * cos_ia_theta));
      F const ry = um2::abs(pitch / (2 * sin_ia_theta));
      F const rb = um2::min(rx, ry);
      F const r = (rb + ring_radii[total_rings - 1]) / 2;
      vertices[num_prev_points + ia] = {r * cos_ia_theta, r * sin_ia_theta};
    }
    // Points on the boundary of the pin (pitch / 2)
    num_prev_points += num_azimuthal;
    for (I ia = 0; ia < 2 * num_azimuthal; ++ia) {
      F sin_ia_gamma = um2::sin(gamma * static_cast<F>(ia));
      F cos_ia_gamma = um2::cos(gamma * static_cast<F>(ia));
      if (um2::abs(sin_ia_gamma) < eps) {
        sin_ia_gamma = 0;
      }
      if (um2::abs(cos_ia_gamma) < eps) {
        cos_ia_gamma = 0;
      }
      F const rx = um2::abs(pitch / (2 * cos_ia_gamma));
      F const ry = um2::abs(pitch / (2 * sin_ia_gamma));
      F const rb = um2::min(rx, ry);
      vertices[num_prev_points + ia] = {rb * cos_ia_gamma, rb * sin_ia_gamma};
    }
    for (I i = 0; i < num_points; ++i) {
      log::debug("Point " + um2::toString(i) + ": " + um2::toString(vertices[i][0]) +
                 ", " + um2::toString(vertices[i][1]));
    }

    //-------------------------------------------------------------------------
    // Get the faces that make up the mesh
    //-------------------------------------------------------------------------
    I const num_faces = na * (nr + 1);
    Vector<Vec<8, I>> faces(num_faces);
    // Triangular quads
    for (I ia = 0; ia < na / 2; ++ia) {
      I const p0 = 0;                   // Center point
      I const p1 = 1 + 2 * na + 4 * ia; // Bottom right point on ring
      I const p2 = p1 + 2;              // Top right point on ring
      I const p3 = 3 + 4 * ia;          // The triangular point
      I p4 = p2 + 2;                    // Top left point on ring
      I const p5 = 1 + 4 * ia;          // Bottom quadratic point
      I const p6 = p1 + 1;              // Right quadratic point
      I const p7 = p3 + 1;              // Top tri quadratic point
      I const p8 = p3 - 1;              // Bottom tri quadratic point
      I const p9 = p2 + 1;              // Top right quadratic point
      I p10 = p7 + 1;                   // Top left quadratic point
      // If we're at the end of the ring, wrap around
      if (p10 == 1 + 2 * na) {
        p4 -= 2 * na;
        p10 -= 2 * na;
      }
      faces[2 * ia] = {p0, p1, p2, p3, p5, p6, p7, p8};
      faces[2 * ia + 1] = {p0, p3, p2, p4, p8, p7, p9, p10};
    }
    // All other faces
    for (I ir = 1; ir < nr + 1; ++ir) {
      I const np = 1 + 2 * na + 3 * na * (ir - 1);
      for (I ia = 0; ia < na; ++ia) {
        I const p0 = np + 2 * ia;
        I const p1 = p0 + 3 * na;
        I p2 = p1 + 2;
        I p3 = p0 + 2;
        I const p4 = np + 2 * na + ia;
        I const p5 = p1 + 1;
        I p6 = p4 + 1;
        I const p7 = p0 + 1;
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
    for (I i = 0; i < num_faces; ++i) {
      LOG_DEBUG("Face ", i, ":", faces[i][0], faces[i][1], faces[i][2], faces[i][3], 
                  faces[i][4], faces[i][5], faces[i][6], faces[i][7]);
    }

    // Shift such that the lower left corner is at the origin
    F const half_pitch = pitch / 2;
    for (I i = 0; i < num_points; ++i) {
      vertices[i] += half_pitch;
      // Fix close to zero values
      if (um2::abs(vertices[i][0]) < eps) {
        vertices[i][0] = 0;
      }
      if (um2::abs(vertices[i][1]) < eps) {
        vertices[i][1] = 0;
      }
    }
    Quad8FVM mesh(vertices, faces);
    //     mesh.populateVF();
    this->_quad8s.push_back(um2::move(mesh));
    LOG_DEBUG("Finished creating mesh");
    return mesh_id;
  }
  log::error("Only linear and quadratic meshes are supported for a cylindrical pin mesh");
  return -1;
}

//=============================================================================
// addRectangularPinMesh
//=============================================================================

auto
SpatialPartition::addRectangularPinMesh(Vec2<F> dxdy, I nx, I ny) -> I
{
  if (dxdy[0] <= 0 || dxdy[1] <= 0) {
    log::error("Pin dimensions must be positive");
  }
  if (nx <= 0 || ny <= 0) {
    log::error("Number of divisions in x and y must be positive");
  }

  I const mesh_id = _quads.size();
  log::info("Making rectangular pin mesh ", mesh_id);

  // Make the vertices
  Vector<Point2> vertices((nx + 1) * (ny + 1));
  F const delta_x = dxdy[0] / static_cast<F>(nx);
  F const delta_y = dxdy[1] / static_cast<F>(ny);
  for (I j = 0; j < ny + 1; ++j) {
    F const y = static_cast<F>(j) * delta_y;
    for (I i = 0; i < nx + 1; ++i) {
      F const x = static_cast<F>(i) * delta_x;
      vertices[j * (nx + 1) + i] = {x, y};
    }
  }
  // Make the faces
  Vector<Vec<4, I>> faces(nx * ny);
  // Left to right, bottom to top
  for (I j = 0; j < ny; ++j) {
    for (I i = 0; i < nx; ++i) {
      faces[j * nx + i] = {(j) * (nx + 1) + i, (j) * (nx + 1) + i + 1,
                           (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i};
    }
  }
  QuadFVM mesh(vertices, faces);
  //  mesh.populateVF();
  _quads.push_back(um2::move(mesh));
  return mesh_id;
}

//=============================================================================
// addCoarseCell
//=============================================================================

auto
SpatialPartition::addCoarseCell(Vec2<F> const dxdy, MeshType const mesh_type,
                                 I const mesh_id,
                                 Vector<MaterialID> const & material_ids) -> I
{
  I const cc_id = _coarse_cells.size();
  log::info("Making coarse cell ", cc_id);
  // Ensure dx and dy are positive
  if (dxdy[0] <= 0 || dxdy[1] <= 0) {
    log::error("dx and dy must be positive");
    return -1;
  }
  // Ensure that the mesh exists
  if (mesh_id != -1) {
    checkMeshExists(mesh_type, mesh_id);
    // Make sure materials are specified
    if (material_ids.empty()) {
      log::error("No materials specified");
      return -1;
    }
  }

  // Create the coarse cell
  _coarse_cells.push_back(CoarseCell{dxdy, mesh_type, mesh_id, material_ids});
  return cc_id;
}

//=============================================================================
// addRTM
//=============================================================================

auto
SpatialPartition::addRTM(Vector<Vector<I>> const & cc_ids) -> I
{
  I const rtm_id = _rtms.size();
  log::info("Making ray tracing module ", rtm_id);
  Vector<I> unique_cc_ids;
  Vector<Vec2<F>> dxdy;
  // Ensure that all coarse cells exist
  I const num_cc = _coarse_cells.size();
  for (auto const & cc_ids_row : cc_ids) {
    for (auto const & id : cc_ids_row) {
      if (id < 0 || id >= num_cc) {
        log::error("Coarse cell ", id, " does not exist");
        return -1;
      }
      auto * const it = std::find(unique_cc_ids.begin(), unique_cc_ids.end(), id);
      if (it == unique_cc_ids.end()) {
        unique_cc_ids.push_back(id);
        // We know id > 0, so subtracting 1 is safe
        dxdy.push_back(_coarse_cells[id].dxdy);
      }
    }
  }
  // For a max pin ID N, the RectilinearGrid constructor needs all dxdy from 0 to N.
  // To get around this requirement, we will renumber the coarse cells to be 0, 1, 2,
  // 3, ..., and then use the renumbered IDs to create the RectilinearGrid.
  Vector<Vector<I>> cc_ids_renumbered(cc_ids.size());
  for (I i = 0; i < cc_ids.size(); ++i) {
    cc_ids_renumbered[i].resize(cc_ids[i].size());
    for (I j = 0; j < cc_ids[i].size(); ++j) {
      auto * const it =
          std::find(unique_cc_ids.begin(), unique_cc_ids.end(), cc_ids[i][j]);
      ASSERT(it != unique_cc_ids.cend());
      cc_ids_renumbered[i][j] = static_cast<I>(it - unique_cc_ids.begin());
    }
  }
  // Create the rectilinear grid
  RectilinearGrid2 const grid(dxdy, cc_ids_renumbered);
  // Ensure the grid has the same dxdy as all other RTMs
  if (!_rtms.empty()) {
    auto constexpr eps = eps_distance;
    if (um2::abs(grid.width() - _rtms[0].grid().width()) > eps ||
        um2::abs(grid.height() - _rtms[0].grid().height()) > eps) {
      log::error("All RTMs must have the same dxdy");
      return -1;
    }
  }
  // Flatten the coarse cell IDs (rows are reversed)
  I const num_rows = cc_ids.size();
  I const num_cols = cc_ids[0].size();
  Vector<I> cc_ids_flat(num_rows * num_cols);
  for (I i = 0; i < num_rows; ++i) {
    for (I j = 0; j < num_cols; ++j) {
      cc_ids_flat[i * num_cols + j] = static_cast<I>(cc_ids[num_rows - 1 - i][j]);
    }
  }
  RTM rtm(grid, cc_ids_flat);
  _rtms.push_back(um2::move(rtm));
  return rtm_id;
}

//=============================================================================
// addLattice
//=============================================================================
////
//// template <std::floating_point T, std::integral I>
//// auto
//// SpatialPartition::stdMakeLattice(std::vector<std::vector<I>> const & rtm_ids)
////    -> I
////{
////  // Convert to um2::Vector
////  Vector<Vector<I>> rtm_ids_um2(static_cast<I>(rtm_ids.size()));
////  for (size_t i = 0; i < rtm_ids.size(); ++i) {
////    rtm_ids_um2[static_cast<I>(i)].resize(static_cast<I>(rtm_ids[i].size()));
////    for (size_t j = 0; j < rtm_ids[i].size(); ++j) {
////      rtm_ids_um2[static_cast<I>(i)][static_cast<I>(j)] =
////          static_cast<I>(rtm_ids[i][j]);
////    }
////  }
////  return addLattice(rtm_ids_um2);
////}

auto
SpatialPartition::addLattice(Vector<Vector<I>> const & rtm_ids) -> I
{
  I const lat_id = _lattices.size();
  log::info("Making lattice ", lat_id);
  // Ensure that all RTMs exist
  I const num_rtm = _rtms.size();
  for (auto const & rtm_ids_row : rtm_ids) {
    auto const * const it =
        std::find_if(rtm_ids_row.begin(), rtm_ids_row.end(),
                     [num_rtm](I const id) { return id < 0 || id >= num_rtm; });
    if (it != rtm_ids_row.cend()) {
      log::error("RTM ", *it, " does not exist");
      return -1;
    }
  }
  // Create the lattice
  // Ensure each row has the same number of columns
  Point2 const minima(0, 0);
  Vec2<F> const spacing = _rtms[0].grid().extents();
  I const num_rows = rtm_ids.size();
  I const num_cols = rtm_ids[0].size();
  for (I i = 1; i < num_rows; ++i) {
    if (rtm_ids[i].size() != num_cols) {
      log::error("Each row must have the same number of columns");
      return -1;
    }
  }
  Vec2<I> const num_cells(num_cols, num_rows);
  RegularGrid2 const grid(minima, spacing, num_cells);
  // Flatten the RTM IDs (rows are reversed)
  Vector<I> rtm_ids_flat(num_rows * num_cols);
  for (I i = 0; i < num_rows; ++i) {
    for (I j = 0; j < num_cols; ++j) {
      rtm_ids_flat[i * num_cols + j] = static_cast<I>(rtm_ids[num_rows - 1 - i][j]);
    }
  }
  Lattice lat(grid, rtm_ids_flat);
  _lattices.push_back(um2::move(lat));
  return lat_id;
}

//=============================================================================
// addAssembly
//=============================================================================

auto
SpatialPartition::addAssembly(Vector<I> const & lat_ids, Vector<F> const & z) -> I
{
  I const asy_id = _assemblies.size();
  log::info("Making assembly ", asy_id);
  // Ensure that all lattices exist
  I const num_lat = _lattices.size();
  {
    auto const * const it =
        std::find_if(lat_ids.cbegin(), lat_ids.cend(),
                     [num_lat](I const id) { return id < 0 || id >= num_lat; });
    if (it != lat_ids.end()) {
      log::error("Lattice " , *it, " does not exist");
      return -1;
    }
  }
  // Ensure the number of lattices is 1 less than the number of z-planes
  if (lat_ids.size() + 1 != z.size()) {
    log::error("The number of lattices must be 1 less than the number of z-planes");
    return -1;
  }
  // Ensure all z-planes are in ascending order
  if (!um2::is_sorted(z.begin(), z.end())) {
    log::error("The z-planes must be in ascending order");
    return -1;
  }
  // Ensure this assembly is the same height as all other assemblies
  if (!_assemblies.empty()) {
    auto constexpr eps = eps_distance;
    auto const assem_top = _assemblies[0].grid().xMax();
    auto const assem_bot = _assemblies[0].grid().xMin();
    if (um2::abs(z.back() - assem_top) > eps || um2::abs(z.front() - assem_bot) > eps) {
      log::error("All assemblies must have the same height");
      return -1;
    }
  }
  // Ensure the lattices all have the same dimensions. Since they are composed of RTMs,
  // it is sufficient to check numXCells and numYCells.
  {
    I const num_xcells = _lattices[lat_ids[0]].grid().numXCells();
    I const num_ycells = _lattices[lat_ids[0]].grid().numYCells();
    auto const * const it = std::find_if(
        lat_ids.cbegin(), lat_ids.cend(), [num_xcells, num_ycells, this](I const id) {
          return this->_lattices[id].grid().numXCells() != num_xcells ||
                 this->_lattices[id].grid().numYCells() != num_ycells;
        });
    if (it != lat_ids.end()) {
      log::error("All lattices must have the same xy-dimensions");
      return -1;
    }
  }

  Vector<I> lat_ids_i(lat_ids.size());
  for (I i = 0; i < lat_ids.size(); ++i) {
    lat_ids_i[i] = static_cast<I>(lat_ids[i]);
  }

  RectilinearGrid1 grid;
  grid.divs(0).resize(z.size());
  um2::copy(z.cbegin(), z.cend(), grid.divs(0).begin());
  Assembly asy(grid, lat_ids_i);
  _assemblies.push_back(um2::move(asy));
  return asy_id;
}

//=============================================================================
// addCore
//=============================================================================

// template <std::floating_point T, std::integral I>
// auto
// SpatialPartition::stdMakeCore(std::vector<std::vector<I>> const & asy_ids)
//    -> I
//{
//  // Convert to um2::Vector
//  Vector<Vector<I>> asy_ids_um2(static_cast<I>(asy_ids.size()));
//  for (size_t i = 0; i < asy_ids.size(); ++i) {
//    asy_ids_um2[static_cast<I>(i)].resize(static_cast<I>(asy_ids[i].size()));
//    for (size_t j = 0; j < asy_ids[i].size(); ++j) {
//      asy_ids_um2[static_cast<I>(i)][static_cast<I>(j)] =
//          static_cast<I>(asy_ids[i][j]);
//    }
//  }
//  return addCore(asy_ids_um2);
//}

auto
SpatialPartition::addCore(Vector<Vector<I>> const & asy_ids) -> I
{
  log::info("Making core");
  // Ensure it is not already made
  if (!_core.children().empty()) {
    log::error("The core has already been made");
    return -1;
  }

  // Ensure that all assemblies exist
  I const num_asy = _assemblies.size();
  for (auto const & asy_ids_row : asy_ids) {
    auto const * const it =
        std::find_if(asy_ids_row.cbegin(), asy_ids_row.cend(),
                     [num_asy](I const id) { return id < 0 || id >= num_asy; });
    if (it != asy_ids_row.end()) {
      log::error("Assembly ", *it, " does not exist");
      return -1;
    }
  }
  Vector<Vec2<F>> dxdy(num_asy);
  for (I i = 0; i < num_asy; ++i) {
    auto const lat_id = _assemblies[i].getChild(0);
    dxdy[i] = _lattices[lat_id].grid().extents();
  }
  // Create the rectilinear grid
  RectilinearGrid2 const grid(dxdy, asy_ids);
  // Flatten the assembly IDs (rows are reversed)
  I const num_rows = asy_ids.size();
  I const num_cols = asy_ids[0].size();
  Vector<I> asy_ids_flat(num_rows * num_cols);
  for (I i = 0; i < num_rows; ++i) {
    if (asy_ids[i].size() != num_cols) {
      log::error("Each row must have the same number of columns");
      return -1;
    }
    for (I j = 0; j < num_cols; ++j) {
      asy_ids_flat[i * num_cols + j] = static_cast<I>(asy_ids[num_rows - 1 - i][j]);
    }
  }
  Core core(grid, asy_ids_flat);
  _core = um2::move(core);
  return 0;
}

//=============================================================================
// importCoarseCells
//=============================================================================

void
SpatialPartition::importCoarseCells(String const & filename)
{
  log::info("Importing coarse cells from ", filename);
  PolytopeSoup mesh_file;
  mesh_file.read(filename);

  // Get the materials
  Vector<String> material_names;
  mesh_file.getMaterialNames(material_names);
  _materials.resize(material_names.size());
  for (I i = 0; i < material_names.size(); ++i) {
    _materials[i].setName(material_names[i].substr(9));
  }

  // For each coarse cell
  std::stringstream ss;
  I const num_coarse_cells = numCoarseCells();
  for (I i = 0; i < num_coarse_cells; ++i) {
    // Get the submesh for the coarse cell
    ss.str("");
    ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << i;
    String const cc_name(ss.str().c_str());
    PolytopeSoup cc_submesh;
    mesh_file.getSubmesh(cc_name, cc_submesh);

    // Get the mesh type and material IDs
    MeshType const mesh_type = cc_submesh.getMeshType();
    CoarseCell & cc = _coarse_cells[i];
    cc.mesh_type = mesh_type;
    Vector<MaterialID> mat_ids;
    cc_submesh.getMaterialIDs(mat_ids, material_names);
    cc.material_ids.resize(mat_ids.size());
    um2::copy(mat_ids.cbegin(), mat_ids.cend(), cc.material_ids.begin());

    // Create the FaceVertexMesh and shift it from global coordinates to local
    // coordinates, with the bottom left corner of the AABB at the origin
    AxisAlignedBox2 bb = AxisAlignedBox2::empty();
    Point2 * vertices = nullptr;
    I const num_verts = cc_submesh.numVerts();
    switch (mesh_type) {
    case MeshType::Tri:
      cc.mesh_id = _tris.size();
      _tris.push_back(um2::move(TriFVM(cc_submesh)));
      bb = _tris.back().boundingBox();
      vertices = _tris.back().vertices().data();
      break;
    case MeshType::Quad:
      cc.mesh_id = _quads.size();
      _quads.push_back(um2::move(QuadFVM(cc_submesh)));
      bb = _quads.back().boundingBox();
      vertices = _quads.back().vertices().data();
      break;
    case MeshType::QuadraticTri:
      cc.mesh_id = _tri6s.size();
      _tri6s.push_back(um2::move(Tri6FVM(cc_submesh)));
      bb = _tri6s.back().boundingBox();
      vertices = _tri6s.back().vertices().data();
      break;
    case MeshType::QuadraticQuad:
      cc.mesh_id = _quad8s.size();
      _quad8s.push_back(um2::move(Quad8FVM(cc_submesh)));
      bb = _quad8s.back().boundingBox();
      vertices = _quad8s.back().vertices().data();
      break;
    default:
      log::error("Mesh type not supported");
    }

    // Shift the points so that the min point is at the origin.
    Point2 const min_point = bb.minima();
    for (I ip = 0; ip < num_verts; ++ip) {
      vertices[ip] -= min_point;
    }
#if UM2_ENABLE_ASSERTS
    Point2 const dxdy = bb.maxima() - bb.minima();
    ASSERT(isApprox(dxdy, cc.dxdy));
#endif
  }
}

//=============================================================================
// fillHierarchy
//=============================================================================

void
SpatialPartition::fillHierarchy()
{
  // Assumes that everything that has been defined fits in 1 of the next higher 
  // hierarchy levels
  
  // Find the first thing we only have 1 of
  if (numCoarseMeshes() == 1) {
    // We need more info. Double check that we have 1 or more coarsecells
    if (numCoarseCells() == 0) {
      log::error("No coarse cells defined");
      return;
    }
  }

  // If we only have 1 coarse cell, we need to add an RTM, unless we already have one
  if (numCoarseCells() == 1 && numRTMs() == 0) {
    I const id = addRTM({{0}});
    if (id != 0) {
      log::error("Failed to add RTM");
      return;
    }
  }

  // If we only have 1 RTM, we need to add a lattice, unless we already have one
  if (numRTMs() == 1 && numLattices() == 0) {
    I const id = addLattice({{0}});
    if (id != 0) {
      log::error("Failed to add lattice");
      return;
    }
  }

  // If we only have 1 lattice, we need to add an assembly, unless we already have one
  if (numLattices() == 1 && numAssemblies() == 0) {
    I const id = addAssembly({0});
    if (id != 0) {
      log::error("Failed to add assembly");
      return;
    }
  }

  // If we only have 1 assembly, we need to add the core
  if (numAssemblies() == 1) {
    I const id = addCore({{0}});
    if (id != 0) {
      log::error("Failed to add core");
      return;
    }
  }
}
////
//////=============================================================================
////// toPolytopeSoup
//////=============================================================================
////
//// template <std::floating_point T, std::integral I>
//// void
////// NOLINTNEXTLINE(readability-function-cognitive-complexity)
//// SpatialPartition::toPolytopeSoup(PolytopeSoup & soup, bool write_kn) const
////{
////   LOG_DEBUG("Converting spatial partition to polytope soup");
////
////   if (core.children.empty()) {
////     log::error("Core has no children");
////     return;
////   }
////   // Allocate counters for each assembly, lattice, etc.
////   Vector<I> asy_found(assemblies.size(), -1);
////   Vector<I> lat_found(lattices.size(), -1);
////   Vector<I> rtm_found(rtms.size(), -1);
////   Vector<I> cc_found(coarse_cells.size(), -1);
////
////   std::stringstream ss;
////   I total_num_faces = 0;
////   LOG_DEBUG("materials.size() = " + toString(materials.size()));
////   Vector<Vector<I>> material_elsets(materials.size());
////   Vector<F> kn_max;
////   Vector<F> kn_mean;
////   Vector<Vector<F>> cc_kns_max(coarse_cells.size());
////   Vector<Vector<F>> cc_kns_mean(coarse_cells.size());
////
////   // We will encode the M by N dimensions of each assembly, lattice,
////   // etc. as elset data.
////   // For each assembly
////   I const nyasy = core.numYCells();
////   I const nxasy = core.numXCells();
////   for (I iyasy = 0; iyasy < nyasy; ++iyasy) {
////     for (I ixasy = 0; ixasy < nxasy; ++ixasy) {
////       I const asy_faces_prev = total_num_faces;
////       auto const asy_id = static_cast<I>(core.getChild(ixasy, iyasy));
////       I const asy_id_ctr = ++asy_found[asy_id];
////       // Get elset name
////       ss.str("");
////       ss << "Assembly_" << std::setw(5) << std::setfill('0') << asy_id << "_"
////          << std::setw(5) << std::setfill('0') << asy_id_ctr;
////       String const asy_name(ss.str().c_str());
////       LOG_DEBUG("Assembly name: " + asy_name);
////       // Get the assembly offset (lower left corner)
////       AxisAlignedBox2<F> const asy_bb = core.getBox(ixasy, iyasy);
////       Point2 const asy_ll = asy_bb.minima; // Lower left corner
////
////       auto const & assembly = assemblies[asy_id];
////       if (assembly.children.empty()) {
////         log::error("Assembly has no children");
////         return;
////       }
////
////       // For each lattice
////       I const nzlat = assembly.numXCells();
////       for (I izlat = 0; izlat < nzlat; ++izlat) {
////         I const lat_faces_prev = total_num_faces;
////         auto const lat_id = static_cast<I>(assembly.getChild(izlat));
////         I const lat_id_ctr = ++lat_found[lat_id];
////         // Get elset name
////         ss.str("");
////         ss << "Lattice_" << std::setw(5) << std::setfill('0') << lat_id << "_"
////            << std::setw(5) << std::setfill('0') << lat_id_ctr;
////         String const lat_name(ss.str().c_str());
////         LOG_DEBUG("Lattice name: " + lat_name);
////         // Get the lattice offset (z direction)
////         // The midplane is the location that the geometry was sampled at.
////         F const low_z = assembly.grid.divs[0][izlat];
////         F const high_z = assembly.grid.divs[0][izlat + 1];
////         F const lat_z = (low_z + high_z) / 2;
////
////         // Get the lattice
////         auto const & lattice = lattices[lat_id];
////         if (lattice.children.empty()) {
////           log::error("Lattice has no children");
////           return;
////         }
////
////         // For each RTM
////         I const nyrtm = lattice.numYCells();
////         I const nxrtm = lattice.numXCells();
////         for (I iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
////           for (I ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {
////             I const rtm_faces_prev = total_num_faces;
////             auto const rtm_id = static_cast<I>(lattice.getChild(ixrtm, iyrtm));
////             I const rtm_id_ctr = ++rtm_found[rtm_id];
////             ss.str("");
////             ss << "RTM_" << std::setw(5) << std::setfill('0') << rtm_id << "_"
////                << std::setw(5) << std::setfill('0') << rtm_id_ctr;
////             String const rtm_name(ss.str().c_str());
////             LOG_DEBUG("RTM name: " + rtm_name);
////             // Get the RTM offset (lower left corner)
////             auto const rtm_bb = lattice.getBox(ixrtm, iyrtm);
////             Point2 const rtm_ll = rtm_bb.minima; // Lower left corner
////
////             // Get the rtm
////             auto const & rtm = rtms[rtm_id];
////             if (rtm.children.empty()) {
////               log::error("RTM has no children");
////               return;
////             }
////
////             I const nycells = rtm.numYCells();
////             I const nxcells = rtm.numXCells();
////             for (I iycell = 0; iycell < nycells; ++iycell) {
////               for (I ixcell = 0; ixcell < nxcells; ++ixcell) {
////                 I const cell_faces_prev = total_num_faces;
////                 auto const & cell_id = static_cast<I>(rtm.getChild(ixcell,
/// iycell)); /                 I const cell_id_ctr = ++cc_found[cell_id]; / ss.str(""); /
/// ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << cell_id / << "_" <<
/// std::setw(5) << std::setfill('0') << cell_id_ctr; /                 String const
/// cell_name(ss.str().c_str()); /                 LOG_DEBUG("Coarse cell name: " +
/// cell_name); /                 // Get the cell offset (lower left corner) / auto const
/// cell_bb = rtm.getBox(ixcell, iycell); /                 Point2 const cell_ll =
/// cell_bb.minima; // Lower left corner
////
////                 // Get the mesh type and id of the coarse cell.
////                 MeshType const mesh_type = coarse_cells[cell_id].mesh_type;
////                 I const mesh_id = coarse_cells[cell_id].mesh_id;
////                 LOG_DEBUG("mesh_id = " + toString(mesh_id));
////                 // Add to material elsets
////                 Vector<MaterialID> const & cell_materials =
////                     coarse_cells[cell_id].material_ids;
////                 LOG_DEBUG("cell_materials.size() = " +
////                 toString(cell_materials.size())); for (I iface = 0; iface <
////                 cell_materials.size(); ++iface) {
////                   auto const mat_id =
//// static_cast<I>(static_cast<uint32_t>(cell_materials[iface])); /
/// material_elsets[mat_id].push_back( /                       static_cast<I>(iface +
/// cell_faces_prev)); /                 }
////
////                 Point2 const * fvm_vertices_begin = nullptr;
////                 Point2 const * fvm_vertices_end = nullptr;
////                 switch (mesh_type) {
////                 case MeshType::Tri:
////                   LOG_DEBUG("Mesh type: Tri");
////                   fvm_vertices_begin = tri[mesh_id].vertices.begin();
////                   fvm_vertices_end = tri[mesh_id].vertices.end();
////                   break;
////                 case MeshType::Quad:
////                   LOG_DEBUG("Mesh type: Quad");
////                   fvm_vertices_begin = quad[mesh_id].vertices.begin();
////                   fvm_vertices_end = quad[mesh_id].vertices.end();
////                   break;
////                 case MeshType::QuadraticTri:
////                   LOG_DEBUG("Mesh type: QuadraticTri");
////                   fvm_vertices_begin = quadratic_tri[mesh_id].vertices.begin();
////                   fvm_vertices_end = quadratic_tri[mesh_id].vertices.end();
////                   break;
////                 case MeshType::QuadraticQuad:
////                   LOG_DEBUG("Mesh type: QuadraticQuad");
////                   fvm_vertices_begin = quadratic_quad[mesh_id].vertices.begin();
////                   fvm_vertices_end = quadratic_quad[mesh_id].vertices.end();
////                   break;
////                 default:
////                   log::error("Unsupported mesh type");
////                   return;
////                 } // switch (mesh_type)
////
////                 // Add each vertex to the PolytopeSoup, offsetting by the
////                 // global xyz offset
////                 auto const num_verts_prev = static_cast<I>(soup.numVerts());
////                 Point2 const xy_offset = cell_ll + rtm_ll + asy_ll;
////                 for (auto it = fvm_vertices_begin; it != fvm_vertices_end; ++it) {
////                   Point2 const p = *it + xy_offset;
////                   soup.addVertex(p[0], p[1], lat_z);
////                 }
////
////                 // Add each face to the PolytopeSoup, offsetting by num_verts_prev
////                 LOG_DEBUG("Adding faces to PolytopeSoup");
////                 switch (mesh_type) {
////                 case MeshType::Tri: {
////                   I const verts_per_face = 3;
////                   VTKElemType const elem_type = VTKElemType::Triangle;
////                   Vector<I> conn(verts_per_face);
////                   LOG_DEBUG("tri[mesh_id].fv.size() = " +
////                             toString(tri[mesh_id].fv.size()));
////                   for (I iface = 0; iface < tri[mesh_id].fv.size(); ++iface) {
////                     auto const & face_conn = tri[mesh_id].fv[iface];
////                     for (I i = 0; i < verts_per_face; ++i) {
////                       conn[i] = face_conn[i] + num_verts_prev;
////                     }
////                     soup.addElement(elem_type, conn);
////                   }
////                   if (write_kn) {
////                     if (cc_kns_max[cell_id].empty()) {
////                       LOG_DEBUG("Computing Knudsen numbers");
////                       for (I iface = 0; iface < tri[mesh_id].fv.size(); ++iface) {
////                         F const mcl = tri[mesh_id].getFace(iface).meanChordLength();
////                         auto const mat_id = static_cast<I>(
////                             static_cast<uint32_t>(cell_materials[iface]));
////                         F const t_max = materials[mat_id].xs.getOneGroupTotalXS(
////                             XSReductionStrategy::Max);
////                         F const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
////                             XSReductionStrategy::Mean);
////                         cc_kns_max[cell_id].push_back(static_cast<F>(1) / (t_max *
////                         mcl)); cc_kns_mean[cell_id].push_back(static_cast<F>(1) /
////                                                        (t_mean * mcl));
////                       }
////                     }
////                     for (auto const & kn : cc_kns_max[cell_id]) {
////                       kn_max.push_back(kn);
////                     }
////                     for (auto const & kn : cc_kns_mean[cell_id]) {
////                       kn_mean.push_back(kn);
////                     }
////                   }
////                 } break;
////                 case MeshType::Quad: {
////                   I const verts_per_face = 4;
////                   VTKElemType const elem_type = VTKElemType::Quad;
////                   Vector<I> conn(verts_per_face);
////                   for (I iface = 0; iface < quad[mesh_id].fv.size(); ++iface) {
////                     auto const & face_conn = quad[mesh_id].fv[iface];
////                     for (I i = 0; i < verts_per_face; ++i) {
////                       conn[i] = face_conn[i] + num_verts_prev;
////                     }
////                     soup.addElement(elem_type, conn);
////                   }
////                   if (write_kn) {
////                     if (cc_kns_max[cell_id].empty()) {
////                       LOG_DEBUG("Computing Knudsen numbers");
////                       for (I iface = 0; iface < quad[mesh_id].fv.size(); ++iface)
///{ /                         LOG_DEBUG("face = " + toString(quad[mesh_id].fv[iface][0])
///+ " /                         " + / toString(quad[mesh_id].fv[iface][1]) + " " + /
/// toString(quad[mesh_id].fv[iface][2]) + " " + / toString(quad[mesh_id].fv[iface][3]));
////                         F const mcl = quad[mesh_id].getFace(iface).meanChordLength();
////                         auto const mat_id = static_cast<I>(
////                             static_cast<uint32_t>(cell_materials[iface]));
////                         F const t_max = materials[mat_id].xs.getOneGroupTotalXS(
////                             XSReductionStrategy::Max);
////                         F const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
////                             XSReductionStrategy::Mean);
////                         cc_kns_max[cell_id].push_back(static_cast<F>(1) / (t_max *
////                         mcl)); cc_kns_mean[cell_id].push_back(static_cast<F>(1) /
////                                                        (t_mean * mcl));
////                       }
////                     }
////                     for (auto const & kn : cc_kns_max[cell_id]) {
////                       kn_max.push_back(kn);
////                     }
////                     for (auto const & kn : cc_kns_mean[cell_id]) {
////                       kn_mean.push_back(kn);
////                     }
////                   }
////                 } break;
////                 case MeshType::QuadraticTri: {
////                   I const verts_per_face = 6;
////                   VTKElemType const elem_type = VTKElemType::QuadraticTriangle;
////                   Vector<I> conn(verts_per_face);
////                   for (I iface = 0; iface < quadratic_tri[mesh_id].fv.size();
////                        ++iface) {
////                     auto const & face_conn = quadratic_tri[mesh_id].fv[iface];
////                     for (I i = 0; i < verts_per_face; ++i) {
////                       conn[i] = face_conn[i] + num_verts_prev;
////                     }
////                     soup.addElement(elem_type, conn);
////                   }
////                   if (write_kn) {
////                     if (cc_kns_max[cell_id].empty()) {
////                       LOG_DEBUG("Computing Knudsen numbers");
////                       for (I iface = 0; iface < quadratic_tri[mesh_id].fv.size();
////                            ++iface) {
////                         F const mcl =
////                             quadratic_tri[mesh_id].getFace(iface).meanChordLength();
////                         auto const mat_id = static_cast<I>(
////                             static_cast<uint32_t>(cell_materials[iface]));
////                         F const t_max = materials[mat_id].xs.getOneGroupTotalXS(
////                             XSReductionStrategy::Max);
////                         F const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
////                             XSReductionStrategy::Mean);
////                         cc_kns_max[cell_id].push_back(static_cast<F>(1) / (t_max *
////                         mcl)); cc_kns_mean[cell_id].push_back(static_cast<F>(1) /
////                                                        (t_mean * mcl));
////                       }
////                     }
////                     for (auto const & kn : cc_kns_max[cell_id]) {
////                       kn_max.push_back(kn);
////                     }
////                     for (auto const & kn : cc_kns_mean[cell_id]) {
////                       kn_mean.push_back(kn);
////                     }
////                   }
////                 } break;
////                 case MeshType::QuadraticQuad: {
////                   I const verts_per_face = 8;
////                   VTKElemType const elem_type = VTKElemType::QuadraticQuad;
////                   Vector<I> conn(verts_per_face);
////                   for (I iface = 0; iface < quadratic_quad[mesh_id].fv.size();
////                        ++iface) {
////                     auto const & face_conn = quadratic_quad[mesh_id].fv[iface];
////                     for (I i = 0; i < verts_per_face; ++i) {
////                       conn[i] = face_conn[i] + num_verts_prev;
////                     }
////                     soup.addElement(elem_type, conn);
////                   }
////                   if (write_kn) {
////                     if (cc_kns_max[cell_id].empty()) {
////                       LOG_DEBUG("Computing Knudsen numbers");
////                       for (I iface = 0; iface < quadratic_quad[mesh_id].fv.size();
////                            ++iface) {
////                         F const mcl =
////                             quadratic_quad[mesh_id].getFace(iface).meanChordLength();
////                         auto const mat_id = static_cast<I>(
////                             static_cast<uint32_t>(cell_materials[iface]));
////                         F const t_max = materials[mat_id].xs.getOneGroupTotalXS(
////                             XSReductionStrategy::Max);
////                         F const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
////                             XSReductionStrategy::Mean);
////                         cc_kns_max[cell_id].push_back(static_cast<F>(1) / (t_max *
////                         mcl)); cc_kns_mean[cell_id].push_back(static_cast<F>(1) /
////                                                        (t_mean * mcl));
////                       }
////                     }
////                     for (auto const & kn : cc_kns_max[cell_id]) {
////                       kn_max.push_back(kn);
////                     }
////                     for (auto const & kn : cc_kns_mean[cell_id]) {
////                       kn_mean.push_back(kn);
////                     }
////                   }
////                 } break;
////                 default:
////                   log::error("Unsupported mesh type");
////                   return;
////                 } // switch (mesh_type)
////                 I const num_faces = soup.numElems() - cell_faces_prev;
////
////                 // Add an elset for the cell
////                 Vector<I> cell_ids(num_faces);
////                 um2::iota(cell_ids.begin(), cell_ids.end(),
////                           static_cast<I>(cell_faces_prev));
////                 soup.addElset(cell_name, cell_ids);
////                 total_num_faces += num_faces;
////
////               } // for (ixcell)
////             }   // for (iycell)
////
////             // Add the RTM elset
////             Vector<I> rtm_ids(total_num_faces - rtm_faces_prev);
////             um2::iota(rtm_ids.begin(), rtm_ids.end(),
/// static_cast<I>(rtm_faces_prev)); /             soup.addElset(rtm_name, rtm_ids); / }
/// // for (ixrtm) /         }   // for (iyrtm)
////
////         // Add the lattice elset
////         Vector<I> lat_ids(total_num_faces - lat_faces_prev);
////         um2::iota(lat_ids.begin(), lat_ids.end(), static_cast<I>(lat_faces_prev));
////         soup.addElset(lat_name, lat_ids);
////       } // for (izlat)
////
////       // Add the assembly elset
////       Vector<I> asy_ids(total_num_faces - asy_faces_prev);
////       um2::iota(asy_ids.begin(), asy_ids.end(), static_cast<I>(asy_faces_prev));
////       soup.addElset(asy_name, asy_ids);
////     } // for (ixasy)
////   }   // for (iyasy)
////
////   // Add the material elsets
////   for (I imat = 0; imat < materials.size(); ++imat) {
////     String const mat_name = "Material_" + String(materials[imat].name.data());
////     soup.addElset(mat_name, material_elsets[imat]);
////   }
////
////   Vector<I> all_ids(total_num_faces);
////   um2::iota(all_ids.begin(), all_ids.end(), static_cast<I>(0));
////   // Add the knudsen number elsets
////   if (write_kn) {
////     soup.addElset("Knudsen_Max", all_ids, kn_max);
////     soup.addElset("Knudsen_Mean", all_ids, kn_mean);
////   }
////
////   soup.sortElsets();
//// } // toPolytopeSoup
////
//////==============================================================================
////// getMaterialNames
//////==============================================================================
////
//// template <std::floating_point T, std::integral I>
//// void
//// SpatialPartition::getMaterialNames(Vector<String> & material_names) const
////{
////  material_names.clear();
////  String const mat_prefix = "Material_";
////  for (auto const & material : materials) {
////    String const mat_suffix(material.name.data());
////    material_names.push_back(mat_prefix + mat_suffix);
////  }
////  std::sort(material_names.begin(), material_names.end());
////} // getMaterialNames
////
//////==============================================================================
////// writeXDMF
//////==============================================================================
////
//// template <std::floating_point T, std::integral I>
//// void
////// NOLINTNEXTLINE
//// SpatialPartition::writeXDMF(String const & filepath, bool write_kn) const
////{
////   log::info("Writing XDMF file: " + filepath);
////
////   // Setup HDF5 file
////   // Get the h5 file name
////   I last_slash = filepath.find_last_of('/');
////   if (last_slash == String::npos) {
////     last_slash = 0;
////   }
////   I const h5filepath_end = last_slash == 0 ? 0 : last_slash + 1;
////   LOG_DEBUG("h5filepath_end: " + toString(h5filepath_end));
////   String const h5filename =
////       filepath.substr(h5filepath_end, filepath.size() - 5 - h5filepath_end) + ".h5";
////   LOG_DEBUG("h5filename: " + h5filename);
////   String const h5filepath = filepath.substr(0, h5filepath_end);
////   LOG_DEBUG("h5filepath: " + h5filepath);
////   H5::H5File h5file((h5filepath + h5filename).c_str(), H5F_ACC_TRUNC);
////
////   // Setup XML file
////   pugi::xml_document xdoc;
////
////   // XDMF root node
////   pugi::xml_node xroot = xdoc.append_child("Xdmf");
////   xroot.append_attribute("Version") = "3.0";
////
////   // Domain node
////   pugi::xml_node xdomain = xroot.append_child("Domain");
////
////   // Get the material names from elset names, in alphabetical order.
////   Vector<String> material_names;
////   getMaterialNames(material_names);
////   std::sort(material_names.begin(), material_names.end());
////
////   // If there are any materials, add an information node listing them
////   if (!material_names.empty()) {
////     pugi::xml_node xinfo = xdomain.append_child("Information");
////     xinfo.append_attribute("Name") = "Materials";
////     String mats;
////     for (I i = 0; i < material_names.size(); ++i) {
////       auto const & mat_name = material_names[i];
////       String const short_name = mat_name.substr(9, mat_name.size() - 9);
////       mats += short_name;
////       if (i + 1 < material_names.size()) {
////         mats += ", ";
////       }
////     }
////     xinfo.append_child(pugi::node_pcdata).set_value(mats.c_str());
////   }
////
////   String const name = h5filename.substr(0, h5filename.size() - 3);
////
////   // Core grid
////   pugi::xml_node xcore_grid = xdomain.append_child("Grid");
////   xcore_grid.append_attribute("Name") = name.c_str();
////   xcore_grid.append_attribute("GridType") = "Tree";
////
////   // h5
////   H5::Group const h5core_group = h5file.createGroup(name.c_str());
////   String const h5core_grouppath = "/" + name;
////
////   // Allocate counters for each assembly, lattice, etc.
////   Vector<I> asy_found(assemblies.size(), -1);
////   Vector<I> lat_found(lattices.size(), -1);
////   Vector<I> rtm_found(rtms.size(), -1);
////   Vector<I> cc_found(coarse_cells.size(), -1);
////
////   std::stringstream ss;
////   Vector<PolytopeSoup> soups(coarse_cells.size());
////   Vector<Vector<F>> cc_kns_max(coarse_cells.size());
////   Vector<Vector<F>> cc_kns_mean(coarse_cells.size());
////
////   if (core.children.empty()) {
////     log::error("Core has no children");
////     return;
////   }
////   I const nyasy = core.numYCells();
////   I const nxasy = core.numXCells();
////   // Core M by N
////   pugi::xml_node xcore_info = xcore_grid.append_child("Information");
////   xcore_info.append_attribute("Name") = "M_by_N";
////   String const core_mn_str = toString(nyasy) + " x " + toString(nxasy);
////   xcore_info.append_child(pugi::node_pcdata).set_value(core_mn_str.c_str());
////   // For each assembly
////   for (I iyasy = 0; iyasy < nyasy; ++iyasy) {
////     for (I ixasy = 0; ixasy < nxasy; ++ixasy) {
////       auto const asy_id = static_cast<I>(core.getChild(ixasy, iyasy));
////       I const asy_id_ctr = ++asy_found[asy_id];
////       // Get elset name
////       ss.str("");
////       ss << "Assembly_" << std::setw(5) << std::setfill('0') << asy_id << "_"
////          << std::setw(5) << std::setfill('0') << asy_id_ctr;
////       String const asy_name(ss.str().c_str());
////       LOG_DEBUG("Assembly name: " + asy_name);
////       // Get the assembly offset (lower left corner)
////       AxisAlignedBox2<F> const asy_bb = core.getBox(ixasy, iyasy);
////       Point2 const asy_ll = asy_bb.minima; // Lower left corner
////
////       auto const & assembly = assemblies[asy_id];
////       if (assembly.children.empty()) {
////         log::error("Assembly has no children");
////         return;
////       }
////
////       // Create the XML grid
////       pugi::xml_node xasy_grid = xcore_grid.append_child("Grid");
////       xasy_grid.append_attribute("Name") = ss.str().c_str();
////       xasy_grid.append_attribute("GridType") = "Tree";
////
////       // Write the M by N information
////       I const nzlat = assembly.numXCells();
////       pugi::xml_node xasy_info = xasy_grid.append_child("Information");
////       xasy_info.append_attribute("Name") = "M_by_N";
////       String const asy_mn_str = toString(nzlat) + " x 1";
////       xasy_info.append_child(pugi::node_pcdata).set_value(asy_mn_str.c_str());
////
////       // Create the h5 group
////       String const h5asy_grouppath = h5core_grouppath + "/" +
/// String(ss.str().c_str()); /       H5::Group const h5asy_group =
/// h5file.createGroup(h5asy_grouppath.c_str());
////
////       // For each lattice
////       for (I izlat = 0; izlat < nzlat; ++izlat) {
////         auto const lat_id = static_cast<I>(assembly.getChild(izlat));
////         I const lat_id_ctr = ++lat_found[lat_id];
////         // Get elset name
////         ss.str("");
////         ss << "Lattice_" << std::setw(5) << std::setfill('0') << lat_id << "_"
////            << std::setw(5) << std::setfill('0') << lat_id_ctr;
////         String const lat_name(ss.str().c_str());
////         LOG_DEBUG("Lattice name: " + lat_name);
////         // Get the lattice offset (z direction)
////         // The midplane is the location that the geometry was sampled at.
////         F const low_z = assembly.grid.divs[0][izlat];
////         F const high_z = assembly.grid.divs[0][izlat + 1];
////         F const lat_z = (low_z + high_z) / 2;
////
////         // Get the lattice
////         auto const & lattice = lattices[lat_id];
////         if (lattice.children.empty()) {
////           log::error("Lattice has no children");
////           return;
////         }
////
////         // Create the XML grid
////         pugi::xml_node xlat_grid = xasy_grid.append_child("Grid");
////         xlat_grid.append_attribute("Name") = ss.str().c_str();
////         xlat_grid.append_attribute("GridType") = "Tree";
////
////         // Add the Z information for the lattice
////         pugi::xml_node xlat_info = xlat_grid.append_child("Information");
////         xlat_info.append_attribute("Name") = "Z";
////         String const z_values =
////             toString(low_z) + ", " + toString(lat_z) + ", " + toString(high_z);
////         xlat_info.append_child(pugi::node_pcdata).set_value(z_values.c_str());
////
////         // Write the M by N information
////         I const nyrtm = lattice.numYCells();
////         I const nxrtm = lattice.numXCells();
////         pugi::xml_node xlat_mn_info = xlat_grid.append_child("Information");
////         xlat_mn_info.append_attribute("Name") = "M_by_N";
////         String const lat_mn_str = toString(nyrtm) + " x " + toString(nxrtm);
////         xlat_mn_info.append_child(pugi::node_pcdata).set_value(lat_mn_str.c_str());
////
////         // Create the h5 group
////         String const h5lat_grouppath = h5asy_grouppath + "/" +
////         String(ss.str().c_str()); H5::Group const h5lat_group =
////         h5file.createGroup(h5lat_grouppath.c_str());
////
////         // For each RTM
////         for (I iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
////           for (I ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {
////             auto const rtm_id = static_cast<I>(lattice.getChild(ixrtm, iyrtm));
////             I const rtm_id_ctr = ++rtm_found[rtm_id];
////             ss.str("");
////             ss << "RTM_" << std::setw(5) << std::setfill('0') << rtm_id << "_"
////                << std::setw(5) << std::setfill('0') << rtm_id_ctr;
////             String const rtm_name(ss.str().c_str());
////             LOG_DEBUG("RTM name: " + rtm_name);
////             // Get the RTM offset (lower left corner)
////             auto const rtm_bb = lattice.getBox(ixrtm, iyrtm);
////             Point2 const rtm_ll = rtm_bb.minima; // Lower left corner
////
////             // Get the rtm
////             auto const & rtm = rtms[rtm_id];
////             if (rtm.children.empty()) {
////               log::error("RTM has no children");
////               return;
////             }
////
////             // Create the XML grid
////             pugi::xml_node xrtm_grid = xlat_grid.append_child("Grid");
////             xrtm_grid.append_attribute("Name") = ss.str().c_str();
////             xrtm_grid.append_attribute("GridType") = "Tree";
////
////             // Write the M by N information
////             I const nycells = rtm.numYCells();
////             I const nxcells = rtm.numXCells();
////             pugi::xml_node xrtm_mn_info = xrtm_grid.append_child("Information");
////             xrtm_mn_info.append_attribute("Name") = "M_by_N";
////             String const rtm_mn_str = toString(nycells) + " x " + toString(nxcells);
//// xrtm_mn_info.append_child(pugi::node_pcdata).set_value(rtm_mn_str.c_str());
////
////             // Create the h5 group
////             String const h5rtm_grouppath =
////                 h5lat_grouppath + "/" + String(ss.str().c_str());
////             H5::Group const h5rtm_group =
/// h5file.createGroup(h5rtm_grouppath.c_str());
////
////             for (I iycell = 0; iycell < nycells; ++iycell) {
////               for (I ixcell = 0; ixcell < nxcells; ++ixcell) {
////                 auto const & cell_id = static_cast<I>(rtm.getChild(ixcell,
/// iycell)); /                 I const cell_id_ctr = ++cc_found[cell_id]; / ss.str(""); /
/// ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << cell_id / << "_" <<
/// std::setw(5) << std::setfill('0') << cell_id_ctr; /                 String const
/// cell_name(ss.str().c_str()); /                 LOG_DEBUG("Coarse cell name: " +
/// cell_name); /                 // Get the cell offset (lower left corner) / auto const
/// cell_bb = rtm.getBox(ixcell, iycell); /                 Point2 const cell_ll =
/// cell_bb.minima; // Lower left corner
////
////                 // Get the mesh type and id of the coarse cell.
////                 MeshType const mesh_type = coarse_cells[cell_id].mesh_type;
////                 I const mesh_id = coarse_cells[cell_id].mesh_id;
////                 LOG_DEBUG("mesh_id = " + toString(mesh_id));
////                 // Add to material elsets
////                 Vector<MaterialID> const & cell_materials =
////                     coarse_cells[cell_id].material_ids;
////                 LOG_DEBUG("cell_materials.size() = " +
////                 toString(cell_materials.size()));
////
////                 // Convert the mesh into PolytopeSoup
////                 PolytopeSoup & soup = soups[cell_id];
////                 if (soups[cell_id].numElems() == 0) {
////                   switch (mesh_type) {
////                   case MeshType::Tri:
////                     LOG_DEBUG("Mesh type: Tri");
////                     tri[mesh_id].toPolytopeSoup(soup);
////                     if (write_kn) {
////                       if (cc_kns_max[cell_id].empty()) {
////                         LOG_DEBUG("Computing Knudsen numbers");
////                         cc_kns_max[cell_id].resize(tri[mesh_id].fv.size());
////                         cc_kns_mean[cell_id].resize(tri[mesh_id].fv.size());
////                         for (I iface = 0; iface < tri[mesh_id].fv.size(); ++iface)
///{ /                           F const mcl =
/// tri[mesh_id].getFace(iface).meanChordLength(); /                           auto const
/// mat_id = static_cast<I>( / static_cast<uint32_t>(cell_materials[iface])); / F const
/// t_max = materials[mat_id].xs.getOneGroupTotalXS( / XSReductionStrategy::Max); / F
/// const t_mean = materials[mat_id].xs.getOneGroupTotalXS( / XSReductionStrategy::Mean);
/// / cc_kns_max[cell_id][iface] = static_cast<F>(1) / (t_max * / mcl);
/// cc_kns_mean[cell_id][iface] = /                               static_cast<F>(1) /
///(t_mean * mcl); /                         } /                       } / } / break; /
/// case MeshType::Quad: /                     LOG_DEBUG("Mesh type: Quad"); /
/// quad[mesh_id].toPolytopeSoup(soup); /                     if (write_kn) { / if
///(cc_kns_max[cell_id].empty()) { /                         LOG_DEBUG("Computing Knudsen
/// numbers"); / cc_kns_max[cell_id].resize(quad[mesh_id].fv.size()); /
/// cc_kns_mean[cell_id].resize(quad[mesh_id].fv.size()); /                         for
///(I iface = 0; iface < quad[mesh_id].fv.size(); ++iface) /                         {
////                           F const mcl =
/// quad[mesh_id].getFace(iface).meanChordLength(); /                           auto const
/// mat_id = static_cast<I>( / static_cast<uint32_t>(cell_materials[iface])); / F const
/// t_max = materials[mat_id].xs.getOneGroupTotalXS( / XSReductionStrategy::Max); / F
/// const t_mean = materials[mat_id].xs.getOneGroupTotalXS( / XSReductionStrategy::Mean);
/// / cc_kns_max[cell_id][iface] = static_cast<F>(1) / (t_max * / mcl);
/// cc_kns_mean[cell_id][iface] = /                               static_cast<F>(1) /
///(t_mean * mcl); /                         } /                       } / } / break; /
/// case MeshType::QuadraticTri: /                     LOG_DEBUG("Mesh type:
/// QuadraticTri"); /                     quadratic_tri[mesh_id].toPolytopeSoup(soup); /
/// if (write_kn) { /                       if (cc_kns_max[cell_id].empty()) { /
/// LOG_DEBUG("Computing Knudsen numbers"); /
/// cc_kns_max[cell_id].resize(quadratic_tri[mesh_id].fv.size()); /
/// cc_kns_mean[cell_id].resize(quadratic_tri[mesh_id].fv.size()); / for (I iface = 0;
/// iface < quadratic_tri[mesh_id].fv.size(); /                              ++iface) { /
/// F const mcl = / quadratic_tri[mesh_id].getFace(iface).meanChordLength(); / auto const
/// mat_id = static_cast<I>( / static_cast<uint32_t>(cell_materials[iface])); / F const
/// t_max = materials[mat_id].xs.getOneGroupTotalXS( / XSReductionStrategy::Max); / F
/// const t_mean = materials[mat_id].xs.getOneGroupTotalXS( / XSReductionStrategy::Mean);
/// / cc_kns_max[cell_id][iface] = static_cast<F>(1) / (t_max * / mcl);
/// cc_kns_mean[cell_id][iface] = /                               static_cast<F>(1) /
///(t_mean * mcl); /                         } /                       } / } / break; /
/// case MeshType::QuadraticQuad: /                     LOG_DEBUG("Mesh type:
/// QuadraticQuad"); /                     quadratic_quad[mesh_id].toPolytopeSoup(soup); /
/// if (write_kn) { /                       if (cc_kns_max[cell_id].empty()) { /
/// LOG_DEBUG("Computing Knudsen numbers"); /
/// cc_kns_max[cell_id].resize(quadratic_quad[mesh_id].fv.size()); /
/// cc_kns_mean[cell_id].resize(quadratic_quad[mesh_id].fv.size()); / for (I iface = 0;
/// iface < quadratic_quad[mesh_id].fv.size(); /                              ++iface) { /
/// F const mcl = / quadratic_quad[mesh_id].getFace(iface).meanChordLength(); / auto const
/// mat_id = static_cast<I>( / static_cast<uint32_t>(cell_materials[iface])); / F const
/// t_max = materials[mat_id].xs.getOneGroupTotalXS( / XSReductionStrategy::Max); / F
/// const t_mean = materials[mat_id].xs.getOneGroupTotalXS( / XSReductionStrategy::Mean);
/// / cc_kns_max[cell_id][iface] = static_cast<F>(1) / (t_max * / mcl);
/// cc_kns_mean[cell_id][iface] = /                               static_cast<F>(1) /
///(t_mean * mcl); /                         } /                       } / } / break; /
/// default: /                     log::error("Unsupported mesh type"); / return; / } //
/// switch (mesh_type)
////
////                   // add Material elsets
////                   I const cc_nfaces = cell_materials.size();
////                   Vector<I> cc_mats(cc_nfaces);
////                   for (I i = 0; i < cc_nfaces; ++i) {
////                     cc_mats[i] =
////                     static_cast<I>(static_cast<uint32_t>(cell_materials[i]));
////                   }
////                   // Get the unique material ids
////                   Vector<I> cc_mats_sorted = cc_mats;
////                   std::sort(cc_mats_sorted.begin(), cc_mats_sorted.end());
////                   auto * it = std::unique(cc_mats_sorted.begin(),
////                   cc_mats_sorted.end()); I const cc_nunique = static_cast<I>(it
///- /                   cc_mats_sorted.begin()); Vector<I> cc_mats_unique(cc_nunique);
/// for /                   (I i = 0; i < cc_nunique; ++i) { / cc_mats_unique[i] =
/// cc_mats_sorted[i]; /                   } /                   // Create a vector with
/// the face ids for each material /                   Vector<Vector<I>>
/// cc_mats_split(cc_nunique); /                   for (I i = 0; i < cc_nfaces; ++i) {
////                     I const mat_id = cc_mats[i];
////                     auto * mat_it =
////                         std::find(cc_mats_unique.begin(), cc_mats_unique.end(),
////                         mat_id);
////                     I const mat_idx =
////                         static_cast<I>(mat_it - cc_mats_unique.begin());
////                     cc_mats_split[mat_idx].push_back(i);
////                   }
////                   // add each material elset
////                   for (I i = 0; i < cc_nunique; ++i) {
////                     I const mat_id = cc_mats_unique[i];
////                     Vector<I> const & mat_faces = cc_mats_split[i];
////                     String const mat_name =
////                         "Material_" + String(materials[mat_id].name.data());
////                     soup.addElset(mat_name, mat_faces);
////                   }
////
////                   if (write_kn) {
////                     Vector<I> all_faces(cc_nfaces);
////                     um2::iota(all_faces.begin(), all_faces.end(), 0);
////                     soup.addElset("Knudsen_Max", all_faces, cc_kns_max[cell_id]);
////                     soup.addElset("Knudsen_Mean", all_faces, cc_kns_mean[cell_id]);
////                     Vector<F> kns_max = cc_kns_max[cell_id];
////                     Vector<F> kns_mean = cc_kns_mean[cell_id];
////                     std::sort(kns_max.begin(), kns_max.end());
////                     std::sort(kns_mean.begin(), kns_mean.end());
////                     F const kn_max_max = kns_max.back();
////                     F const kn_mean_max = kns_mean.back();
////                     F const kn_max_min = kns_max.front();
////                     F const kn_mean_min = kns_mean.front();
////                     F const kn_max_mean = um2::mean(kns_max.begin(), kns_max.end());
////                     F const kn_mean_mean = um2::mean(kns_mean.begin(),
/// kns_mean.end()); /                     LOG_INFO("Coarse Cell " + toString(cell_id) + "
///" + /                              toString(kn_max_max) + " " + toString(kn_max_min) +
///" " + /                              toString(kn_max_mean)); / LOG_INFO("Coarse Cell "
///+ toString(cell_id) + " " + /                              toString(kn_mean_max) + " "
///+ toString(kn_mean_min) + " " /                              + toString(kn_mean_mean));
////                   }
////                 }
////
////                 // Shift the mesh to global coordinates
////                 Point2 const xy_offset = cell_ll + rtm_ll + asy_ll;
////                 Point3<F> const shift = Point3<F>(xy_offset[0], xy_offset[1], lat_z);
////                 soup.translate(shift);
////
////                 // Write the mesh
////                 soup.writeXDMFUniformGrid(cell_name, material_names, xrtm_grid,
/// h5file, /                                           h5filename, h5rtm_grouppath);
////
////                 // Shift the mesh back to local coordinates
////                 soup.translate(-shift);
////               } // for (ixcell)
////             }   // for (iycell)
////           }     // for (ixrtm)
////         }       // for (iyrtm)
////       }         // for (izlat)
////     }           // for (ixasy)
////   }             // for (iyasy)
////
////   // Write the XML file
////   xdoc.save_file(filepath.c_str(), "  ");
////
////   // Close the HDF5 file
////   h5file.close();
//// } // writeXDMF
////
//////==============================================================================
////// write
//////==============================================================================
////
//// template <std::floating_point T, std::integral I>
//// void
//// SpatialPartition::write(String const & filename, bool write_kn) const
////{
////  if (filename.ends_with(".xdmf")) {
////    writeXDMF(filename, write_kn);
////  } else {
////    log::error("Unsupported file format.");
////  }
////}
//
} // namespace um2::mpact
