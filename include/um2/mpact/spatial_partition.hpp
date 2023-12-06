#pragma once

#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/mesh/rectilinear_partition.hpp>
#include <um2/mesh/regular_partition.hpp>
#include <um2/physics/material.hpp>

#include <iomanip>

namespace um2::mpact
{

//==============================================================================
// MPACT SPATIAL PARTITON
//==============================================================================
//
// An equivalent representation to the various mesh hierarchies in an MPACT model.
//
//  ************************
//  *****VERY IMPORTANT*****
//  ************************
//  - The pin mesh coordinate system origin in MPACT is the center of the pin. Here
//    we use the bottom left corner of the pin mesh as the origin.
//  - In MPACT, two pins with the same mesh but different heights are considered
//    different meshes. Here we consider them the same mesh.
//
// The MPACT spatial partition consists of:
//      1. Core
//          A rectilinear partition of the XY-domain into assemblies. The assemblies
//          must have the same start and stop heights.
//      2. Assembly
//          A rectilinear partition of the Z-domain into 2D axial slices (lattices).
//      3. Lattice
//          A regular partition of the XY-domain into equal-sized axis-aligned
//          rectangles, also known as "ray tracing modules" (RTMs).
//          Each lattice has a local coordinate system with (0, 0) in the bottom
//          left corner.
//      4. RTM
//          A rectilinear partition of the XY-domain into coarse cells.
//          Every RTM is exactly the same width and height in all lattices.
//          This property is a necessity for modular ray tracing.
//          Each RTM has a local coordinate system with (0, 0) in the bottom
//          left corner.
//      5. Coarse cell
//          A 2D axis-aligned box (AABB), containing a mesh which completely
//          fills the box's interior. This mesh is the "fine mesh". It is made
//          up of fine cells (triangles, quadrilaterals, etc.). Each of these
//          fine cells has an integer material ID. This structure is
//          represented as a fine mesh ID and a material ID list ID, allowing the
//          same mesh to be reused for multiple pins with different materials.
//          Each coarse cell has a local coordinate system with (0, 0) in the
//          bottom left corner.
//
//          In MPACT, the coarse cells typically contain the geometry for a single
//          pin, centered in middle of the coarse cell - hence the name "pin cell".
//          In this code, due to the arbitrary nature of the geometry, the coarse
//          cells may contain a piece of a pin, multiple pins, or any other
//          arbitrary geometry.
//

template <std::floating_point T, std::integral I>
struct SpatialPartition {

  struct CoarseCell {
    Vec2<T> dxdy; // dx, dy
    MeshType mesh_type = MeshType::None;
    Size mesh_id = -1;               // index into the corresponding mesh array
    Vector<MaterialID> material_ids; // size = mesh.numFaces()

    PURE [[nodiscard]] constexpr auto
    numFaces() const noexcept -> Size
    {
      return material_ids.size();
    }
  };
  using RTM = RectilinearPartition2<T, I>;
  using Lattice = RegularPartition2<T, I>;
  using Assembly = RectilinearPartition1<T, I>;

  // The children IDs are used to index the corresponding array.
  // Child ID = -1 indicates that the child does not exist. This is used
  // for when the child should be generated automatically.

  RectilinearPartition2<T, I> core;
  Vector<Assembly> assemblies;
  Vector<Lattice> lattices;
  Vector<RTM> rtms;
  Vector<CoarseCell> coarse_cells;

  Vector<Material<T>> materials;

  Vector<TriMesh<2, T, I>> tri;
  Vector<QuadMesh<2, T, I>> quad;
  Vector<QuadraticTriMesh<2, T, I>> quadratic_tri;
  Vector<QuadraticQuadMesh<2, T, I>> quadratic_quad;

  //============================================================================
  // Constructors
  //============================================================================

  constexpr SpatialPartition() noexcept = default;

  //============================================================================
  // Accessors
  //============================================================================

  PURE [[nodiscard]] constexpr auto
  numCoarseCells() const noexcept -> Size;

  PURE [[nodiscard]] constexpr auto
  numRTMs() const noexcept -> Size;

  PURE [[nodiscard]] constexpr auto
  numLattices() const noexcept -> Size;

  PURE [[nodiscard]] constexpr auto
  numAssemblies() const noexcept -> Size;

  //============================================================================
  // Methods
  //============================================================================

  HOSTDEV constexpr void
  clear() noexcept;

  inline void
  checkMeshExists(MeshType mesh_type, Size mesh_id) const;

  auto
  makeCylindricalPinMesh(Vector<T> const & radii, T pitch, Vector<Size> const & num_rings,
                         Size num_azimuthal, Size mesh_order = 1) -> Size;

  auto
  makeRectangularPinMesh(Vec2<T> dxdy, Size nx, Size ny);

  auto
  makeCoarseCell(Vec2<T> dxdy, MeshType mesh_type = MeshType::None, Size mesh_id = -1,
                 Vector<MaterialID> const & material_ids = {}) -> Size;

  auto
  makeRTM(Vector<Vector<Size>> const & cc_ids) -> Size;

  auto
  makeLattice(Vector<Vector<Size>> const & rtm_ids) -> Size;

  auto
  stdMakeLattice(std::vector<std::vector<Size>> const & rtm_ids) -> Size;

  auto
  makeAssembly(Vector<Size> const & lat_ids, Vector<T> const & z = {-1, 1}) -> Size;

  auto
  makeCore(Vector<Vector<Size>> const & asy_ids) -> Size;

  auto
  stdMakeCore(std::vector<std::vector<Size>> const & asy_ids) -> Size;

  void
  importCoarseCells(String const & filename);

  void
  toPolytopeSoup(PolytopeSoup<T, I> & soup, bool write_kn = false) const;

  void
  getMaterialNames(Vector<String> & material_names) const;

  void
  write(String const & filename, bool write_kn = false) const;

  void
  writeXDMF(String const & filepath, bool write_kn = false) const;

}; // struct SpatialPartition

//=============================================================================
// Accessors
//=============================================================================

template <std::floating_point T, std::integral I>
PURE [[nodiscard]] constexpr auto
SpatialPartition<T, I>::numCoarseCells() const noexcept -> Size
{
  return coarse_cells.size();
}

template <std::floating_point T, std::integral I>
PURE [[nodiscard]] constexpr auto
SpatialPartition<T, I>::numRTMs() const noexcept -> Size
{
  return rtms.size();
}

template <std::floating_point T, std::integral I>
PURE [[nodiscard]] constexpr auto
SpatialPartition<T, I>::numLattices() const noexcept -> Size
{
  return lattices.size();
}

template <std::floating_point T, std::integral I>
PURE [[nodiscard]] constexpr auto
SpatialPartition<T, I>::numAssemblies() const noexcept -> Size
{
  return assemblies.size();
}

//=============================================================================
// clear
//=============================================================================

template <std::floating_point T, std::integral I>
HOSTDEV constexpr void
SpatialPartition<T, I>::clear() noexcept
{
  core.clear();
  assemblies.clear();
  lattices.clear();
  rtms.clear();
  coarse_cells.clear();

  materials.clear();

  tri.clear();
  quad.clear();
  quadratic_tri.clear();
  quadratic_quad.clear();
}

//=============================================================================
// makeCylindricalPinMesh
//=============================================================================

template <std::floating_point T, std::integral I>
auto
// NOLINTNEXTLINE
SpatialPartition<T, I>::makeCylindricalPinMesh(Vector<T> const & radii, T const pitch,
                                               Vector<Size> const & num_rings,
                                               Size const num_azimuthal,
                                               Size const mesh_order) -> Size
{
  LOG_DEBUG("Making cylindrical pin mesh");
  if ((num_azimuthal & (num_azimuthal - 1)) != 0) {
    Log::error("The number of azimuthal divisions must be a power of 2");
    return -1;
  }
  if (num_azimuthal < 8) {
    Log::error("The number of azimuthal divisions must be at least 8");
    return -1;
  }
  if (radii.size() != num_rings.size()) {
    Log::error("The number of radii must match the size of num_rings");
    return -1;
  }
  if (std::any_of(radii.begin(), radii.end(),
                  [pitch](double r) { return r > pitch / 2; })) {
    Log::error("The radii must be less than half the pitch");
    return -1;
  }

  Size mesh_id = -1;
  if (mesh_order == 1) {
    mesh_id = this->quad.size();
    Log::info("Making linear quadrilateral cylindrical pin mesh " + toString(mesh_id));
  } else if (mesh_order == 2) {
    mesh_id = this->quadratic_quad.size();
    Log::info("Making quadratic quadrilateral cylindrical pin mesh " + toString(mesh_id));
  } else {
    Log::error("Invalid mesh order");
    return -1;
  }

  // radial region = region containing different materials (rings + outside of
  // the last radius)
  //
  // radial_region_areas = area of each radial region, including outside of the last
  // radius
  //
  // ring = an equal area division of a radial region containing the same material
  //
  // ring_radii = the radius of each ring, NOT including the outside of the last
  // radius
  //   ring_areas = the area of each ring, including the outside of the last radius

  //---------------------------------------------------------------------------
  // Get the area of each radial region (rings + outside of the last ring)
  //---------------------------------------------------------------------------
  Size const num_radial_regions = radii.size() + 1;
  Vector<T> radial_region_areas(num_radial_regions);
  // A0 = pi * r0^2
  // Ai = pi * (ri^2 - ri-1^2)
  radial_region_areas[0] = pi<T> * radii[0] * radii[0];
  for (Size i = 1; i < num_radial_regions - 1; ++i) {
    radial_region_areas[i] = pi<T> * (radii[i] * radii[i] - radii[i - 1] * radii[i - 1]);
  }
  radial_region_areas[num_radial_regions - 1] =
      pitch * pitch - radial_region_areas[num_radial_regions - 2];

  //---------------------------------------------------------------------------
  // Get the radii and areas of each ring after splitting the radial regions
  // This includes outside of the last ring
  //---------------------------------------------------------------------------
  Size const total_rings = std::reduce(num_rings.begin(), num_rings.end(), 0);
  Vector<T> ring_radii(total_rings);
  Vector<T> ring_areas(total_rings + 1);
  // Inside the innermost region
  ring_areas[0] = radial_region_areas[0] / num_rings[0];
  ring_radii[0] = um2::sqrt(ring_areas[0] / pi<T>);
  for (Size i = 1; i < num_rings[0]; ++i) {
    ring_areas[i] = ring_areas[0];
    ring_radii[i] =
        um2::sqrt(ring_areas[i] / pi<T> + ring_radii[i - 1] * ring_radii[i - 1]);
  }
  Size ctr = num_rings[0];
  for (Size ireg = 1; ireg < num_radial_regions - 1; ++ireg) {
    Size const num_rings_in_region = num_rings[ireg];
    T const area_per_ring = radial_region_areas[ireg] / num_rings_in_region;
    for (Size iring = 0; iring < num_rings_in_region; ++iring, ++ctr) {
      ring_areas[ctr] = area_per_ring;
      ring_radii[ctr] =
          um2::sqrt(area_per_ring / pi<T> + ring_radii[ctr - 1] * ring_radii[ctr - 1]);
    }
  }
  // Outside of the last ring
  ring_areas[ctr] = pitch * pitch - pi<T> * ring_radii.back() * ring_radii.back();
  // Log the radii and areas in debug mode
  for (Size i = 0; i < total_rings; ++i) {
    LOG_TRACE("Ring " + toString(i) + " radius: " + toString(ring_radii[i]));
    LOG_TRACE("Ring " + toString(i) + " area: " + toString(ring_areas[i]));
  }
  LOG_TRACE("The area outside of the last ring is " + toString(ring_areas[ctr]));
  // Ensure the sum of the ring areas is equal to pitch^2
  T const sum_ring_areas =
      std::reduce(ring_areas.begin(), ring_areas.end(), static_cast<T>(0));
  ASSERT_NEAR(sum_ring_areas, pitch * pitch, static_cast<T>(1e-6));
  if (mesh_order == 1) {
    // Get the equivalent radius of each ring if it were a quadrilateral
    T const theta = 2 * pi<T> / num_azimuthal;
    T const sin_theta = um2::sin(theta);
    Vector<T> eq_radii(total_rings);
    // The innermost radius is a special case, and is essentially a triangle.
    // A_t = l² * sin(θ) / 2
    // A_ring = num_azi * A_t = l² * sin(θ) * num_azi / 2
    // l = sqrt(2 * A_ring / (sin(θ) * num_azi))
    eq_radii[0] = um2::sqrt(2 * ring_areas[0] / (sin_theta * num_azimuthal));
    // A_q = (l² - l²₀) * sin(θ) / 2
    // A_ring = num_azi * A_q = (l² - l²₀) * sin(θ) * num_azi / 2
    // l = sqrt(2 * A_ring / (sin(θ) * num_azi) + l²₀)
    for (Size i = 1; i < total_rings; ++i) {
      eq_radii[i] = um2::sqrt(2 * ring_areas[i] / (sin_theta * num_azimuthal) +
                              eq_radii[i - 1] * eq_radii[i - 1]);
    }
    for (Size i = 0; i < total_rings; ++i) {
      LOG_TRACE("Ring " + toString(i) + " equivalent radius: " + toString(eq_radii[i]));
    }
    // If any of the equivalent radii are larger than half the pitch, error
    if (std::any_of(eq_radii.begin(), eq_radii.end(),
                    [pitch](double r) { return r > pitch / 2; })) {
      Log::error("The equivalent radius of a ring is larger than half the pitch");
      return -1;
    }
    // Sanity check: ensure the sum of the quadrilateral areas in a ring is equal to
    // the ring area
    ASSERT_NEAR(eq_radii[0] * eq_radii[0] * sin_theta / 2, ring_areas[0] / num_azimuthal,
                static_cast<T>(1e-4));
    for (Size i = 1; i < total_rings; ++i) {
      T const area =
          (eq_radii[i] * eq_radii[i] - eq_radii[i - 1] * eq_radii[i - 1]) * sin_theta / 2;
      ASSERT_NEAR(area, ring_areas[i] / num_azimuthal, static_cast<T>(1e-4));
    }

    //------------------------------------------------------------------------
    // Get the points that make up the mesh
    //------------------------------------------------------------------------
    // The number of points is:
    //   Center point
    //   num_azimuthal / 2, for the points in the innermost ring to make the quads
    //      "triangular"
    //   (num_rings + 1) * num_azimuthal
    Size const num_points = 1 + (total_rings + 1) * num_azimuthal + num_azimuthal / 2;
    Vector<Point2<T>> vertices(num_points);
    LOG_TRACE("The number of points is " + toString(num_points));
    // Center point
    vertices[0] = {0, 0};
    // Triangular points
    LOG_TRACE("Computing the triangular points");
    T const rt = eq_radii[0] / 2;
    for (Size ia = 0; ia < num_azimuthal / 2; ++ia) {
      T const sin_ia_theta = um2::sin(theta * (2 * ia + 1));
      T const cos_ia_theta = um2::cos(theta * (2 * ia + 1));
      vertices[1 + ia] = {rt * cos_ia_theta, rt * sin_ia_theta};
    }
    LOG_TRACE("Computing the quadrilateral points");
    // Quadrilateral points
    // Points on rings, not including the boundary of the pin (pitch / 2 box)
    for (Size ir = 0; ir < total_rings; ++ir) {
      Size const num_prev_points = 1 + num_azimuthal / 2 + ir * num_azimuthal;
      for (Size ia = 0; ia < num_azimuthal; ++ia) {
        T sin_ia_theta = um2::sin(theta * ia);
        T cos_ia_theta = um2::cos(theta * ia);
        if (um2::abs(sin_ia_theta) < static_cast<T>(1e-6)) {
          sin_ia_theta = 0;
        }
        if (um2::abs(cos_ia_theta) < static_cast<T>(1e-6)) {
          cos_ia_theta = 0;
        }
        vertices[num_prev_points + ia] = {eq_radii[ir] * cos_ia_theta,
                                          eq_radii[ir] * sin_ia_theta};
      }
    }
    LOG_TRACE("Computing the boundary points");
    // Points on the boundary of the pin (pitch / 2)
    Size const num_prev_points = 1 + num_azimuthal / 2 + total_rings * num_azimuthal;
    for (Size ia = 0; ia < num_azimuthal; ++ia) {
      T sin_ia_theta = std::sin(theta * ia);
      T cos_ia_theta = std::cos(theta * ia);
      if (um2::abs(sin_ia_theta) < 1e-6) {
        sin_ia_theta = 0;
      }
      if (um2::abs(cos_ia_theta) < 1e-6) {
        cos_ia_theta = 0;
      }
      T const rx = um2::abs(pitch / (2 * cos_ia_theta));
      T const ry = um2::abs(pitch / (2 * sin_ia_theta));
      T const rb = um2::min(rx, ry);
      vertices[num_prev_points + ia] = {rb * cos_ia_theta, rb * sin_ia_theta};
    }
    for (Size i = 0; i < num_points; ++i) {
      LOG_TRACE("Point " + toString(i) + ": " + toString(vertices[i][0]) + ", " +
                toString(vertices[i][1]));
    }

    //------------------------------------------------------------------------
    // Get the faces that make up the mesh
    //------------------------------------------------------------------------
    Size const num_faces = num_azimuthal * (total_rings + 1);
    Vector<Vec<4, I>> faces(num_faces);
    // Establish a few aliases
    Size const na = num_azimuthal;
    Size const nr = total_rings;
    Size const ntric = 1 + na / 2; // Number of triangular points + center point
    // Triangular quads
    for (Size ia = 0; ia < na / 2; ++ia) {
      Size const p0 = 0;                  // Center point
      Size const p1 = ntric + ia * 2;     // Bottom right point on ring
      Size const p2 = ntric + ia * 2 + 1; // Top right point on ring
      Size const p3 = 1 + ia;             // The triangular point
      Size p4 = ntric + ia * 2 + 2;       // Top left point on ring
      // If we're at the end of the ring, wrap around
      if (p4 == ntric + na) {
        p4 = ntric;
      }
      faces[2 * ia] = {p0, p1, p2, p3};
      faces[2 * ia + 1] = {p0, p3, p2, p4};
    }
    // Non-boundary and boundary quads
    for (Size ir = 1; ir < nr + 1; ++ir) {
      for (Size ia = 0; ia < na; ++ia) {
        Size const p0 = ntric + (ir - 1) * na + ia; // Bottom left point
        Size const p1 = ntric + (ir)*na + ia;       // Bottom right point
        Size p2 = ntric + (ir)*na + ia + 1;         // Top right point
        Size p3 = ntric + (ir - 1) * na + ia + 1;   // Top left point
        // If we're at the end of the ring, wrap around
        if (ia + 1 == na) {
          p2 -= na;
          p3 -= na;
        }
        faces[ir * na + ia] = {p0, p1, p2, p3};
      }
    }
    // Shift such that the lower left corner is at the origin
    T const half_pitch = pitch / 2;
    for (Size i = 0; i < num_points; ++i) {
      vertices[i] += half_pitch;
      // Fix close to zero values
      if (um2::abs(vertices[i][0]) < static_cast<T>(1e-6)) {
        vertices[i][0] = 0;
      }
      if (um2::abs(vertices[i][1]) < static_cast<T>(1e-6)) {
        vertices[i][1] = 0;
      }
    }
    QuadMesh<2, T, I> mesh;
    mesh.vertices = vertices;
    mesh.fv = faces;
    mesh.populateVF();
    this->quad.push_back(um2::move(mesh));
    LOG_TRACE("Finished creating mesh");
    return mesh_id;
  }
  if (mesh_order == 2) {
    // Get the equivalent radius of each ring if it were a quadratic quadrilateral
    T const theta = 2 * pi<T> / num_azimuthal;
    T const gamma = theta / 2;
    T const sin_gamma = um2::sin(gamma);
    T const cos_gamma = um2::cos(gamma);
    T const sincos_gamma = sin_gamma * cos_gamma;
    Vector<T> eq_radii(total_rings);
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
    T tri_area = ring_radii[0] * ring_radii[0] * sincos_gamma;
    T ring_area = ring_areas[0] / num_azimuthal;
    T const l0 = ring_radii[0];
    eq_radii[0] = 0.75 * (ring_area - tri_area) / (l0 * sin_gamma) + l0 * cos_gamma;
    for (Size i = 1; i < total_rings; ++i) {
      T const l_im1 = ring_radii[i - 1];
      T const ll_im1 = eq_radii[i - 1];
      T const a_edge_im1 = (4.0 / 3.0) * l_im1 * sin_gamma * (ll_im1 - l_im1 * cos_gamma);
      T const l = ring_radii[i];
      T const a_quad = (l * l - l_im1 * l_im1) * sincos_gamma;
      T const a_ring = ring_areas[i] / num_azimuthal;
      eq_radii[i] =
          0.75 * (a_ring - a_quad + a_edge_im1) / (l * sin_gamma) + l * cos_gamma;
    }
    // Log the equivalent radii in debug mode
    for (Size i = 0; i < total_rings; ++i) {
      Log::debug("Ring " + toString(i) + " equivalent radius: " + toString(eq_radii[i]));
    }
    // If any of the equivalent radii are larger than half the pitch, error
    if (std::any_of(eq_radii.begin(), eq_radii.end(),
                    [pitch](double r) { return r > pitch / 2; })) {
      Log::error("The equivalent radius of a ring is larger than half the pitch.");
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
    Size const na = num_azimuthal;
    Size const nr = total_rings;
    Size const num_points = 1 + 4 * na + 3 * na * nr;
    Vector<Point2<T>> vertices(num_points);
    // Center point
    vertices[0] = {0, 0};
    // Triangular points
    T const rt = ring_radii[0] / 2;
    for (Size ia = 0; ia < na; ++ia) {
      T const sin_ia_theta = um2::sin(ia * theta);
      T const cos_ia_theta = um2::cos(ia * theta);
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
    Size num_prev_points = 1 + 2 * na;
    for (Size ia = 0; ia < 2 * na; ++ia) {
      T sin_ia_gamma = um2::sin(ia * gamma);
      T cos_ia_gamma = um2::cos(ia * gamma);
      if (um2::abs(sin_ia_gamma) < 1e-6) {
        sin_ia_gamma = 0;
      }
      if (um2::abs(cos_ia_gamma) < 1e-6) {
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
    for (Size ir = 1; ir < total_rings; ++ir) {
      num_prev_points = 1 + 4 * na + 3 * na * (ir - 1);
      // Between the rings
      for (Size ia = 0; ia < num_azimuthal; ++ia) {
        T sin_ia_theta = um2::sin(ia * theta);
        T cos_ia_theta = um2::cos(ia * theta);
        if (um2::abs(sin_ia_theta) < 1e-6) {
          sin_ia_theta = 0;
        }
        if (um2::abs(cos_ia_theta) < 1e-6) {
          cos_ia_theta = 0;
        }
        T const r = (ring_radii[ir] + ring_radii[ir - 1]) / 2;
        vertices[num_prev_points + ia] = {r * cos_ia_theta, r * sin_ia_theta};
      }
      num_prev_points += num_azimuthal;
      for (Size ia = 0; ia < 2 * num_azimuthal; ++ia) {
        T sin_ia_gamma = um2::sin(ia * gamma);
        T cos_ia_gamma = um2::cos(ia * gamma);
        if (um2::abs(sin_ia_gamma) < 1e-6) {
          sin_ia_gamma = 0;
        }
        if (um2::abs(cos_ia_gamma) < 1e-6) {
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
    for (Size ia = 0; ia < num_azimuthal; ++ia) {
      T sin_ia_theta = um2::sin(ia * theta);
      T cos_ia_theta = um2::cos(ia * theta);
      if (um2::abs(sin_ia_theta) < 1e-6) {
        sin_ia_theta = 0;
      }
      if (um2::abs(cos_ia_theta) < 1e-6) {
        cos_ia_theta = 0;
      }
      // pitch and last ring radius
      T const rx = um2::abs(pitch / (2 * cos_ia_theta));
      T const ry = um2::abs(pitch / (2 * sin_ia_theta));
      T const rb = um2::min(rx, ry);
      T const r = (rb + ring_radii[total_rings - 1]) / 2;
      vertices[num_prev_points + ia] = {r * cos_ia_theta, r * sin_ia_theta};
    }
    // Points on the boundary of the pin (pitch / 2)
    num_prev_points += num_azimuthal;
    for (Size ia = 0; ia < 2 * num_azimuthal; ++ia) {
      T sin_ia_gamma = um2::sin(gamma * ia);
      T cos_ia_gamma = um2::cos(gamma * ia);
      if (um2::abs(sin_ia_gamma) < 1e-6) {
        sin_ia_gamma = 0;
      }
      if (um2::abs(cos_ia_gamma) < 1e-6) {
        cos_ia_gamma = 0;
      }
      T const rx = um2::abs(pitch / (2 * cos_ia_gamma));
      T const ry = um2::abs(pitch / (2 * sin_ia_gamma));
      T const rb = um2::min(rx, ry);
      vertices[num_prev_points + ia] = {rb * cos_ia_gamma, rb * sin_ia_gamma};
    }
    for (Size i = 0; i < num_points; ++i) {
      Log::debug("Point " + um2::toString(i) + ": " + um2::toString(vertices[i][0]) +
                 ", " + um2::toString(vertices[i][1]));
    }

    //-------------------------------------------------------------------------
    // Get the faces that make up the mesh
    //-------------------------------------------------------------------------
    Size const num_faces = na * (nr + 1);
    Vector<Vec<8, I>> faces(num_faces);
    // Triangular quads
    for (Size ia = 0; ia < na / 2; ++ia) {
      Size const p0 = 0;                   // Center point
      Size const p1 = 1 + 2 * na + 4 * ia; // Bottom right point on ring
      Size const p2 = p1 + 2;              // Top right point on ring
      Size const p3 = 3 + 4 * ia;          // The triangular point
      Size p4 = p2 + 2;                    // Top left point on ring
      Size const p5 = 1 + 4 * ia;          // Bottom quadratic point
      Size const p6 = p1 + 1;              // Right quadratic point
      Size const p7 = p3 + 1;              // Top tri quadratic point
      Size const p8 = p3 - 1;              // Bottom tri quadratic point
      Size const p9 = p2 + 1;              // Top right quadratic point
      Size p10 = p7 + 1;                   // Top left quadratic point
      // If we're at the end of the ring, wrap around
      if (p10 == 1 + 2 * na) {
        p4 -= 2 * na;
        p10 -= 2 * na;
      }
      faces[2 * ia] = {p0, p1, p2, p3, p5, p6, p7, p8};
      faces[2 * ia + 1] = {p0, p3, p2, p4, p8, p7, p9, p10};
    }
    // All other faces
    for (Size ir = 1; ir < nr + 1; ++ir) {
      Size const np = 1 + 2 * na + 3 * na * (ir - 1);
      for (Size ia = 0; ia < na; ++ia) {
        Size const p0 = np + 2 * ia;
        Size const p1 = p0 + 3 * na;
        Size p2 = p1 + 2;
        Size p3 = p0 + 2;
        Size const p4 = np + 2 * na + ia;
        Size const p5 = p1 + 1;
        Size p6 = p4 + 1;
        Size const p7 = p0 + 1;
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
    for (Size i = 0; i < num_faces; ++i) {
      Log::debug("Face " + um2::toString(i) + ": " + um2::toString(faces[i][0]) + ", " +
                 um2::toString(faces[i][1]) + ", " + um2::toString(faces[i][2]) + ", " +
                 um2::toString(faces[i][3]) + ", " + um2::toString(faces[i][4]) + ", " +
                 um2::toString(faces[i][5]) + ", " + um2::toString(faces[i][6]) + ", " +
                 um2::toString(faces[i][7]));
    }

    // Shift such that the lower left corner is at the origin
    T const half_pitch = pitch / 2;
    for (Size i = 0; i < num_points; ++i) {
      vertices[i] += half_pitch;
      // Fix close to zero values
      if (um2::abs(vertices[i][0]) < 1e-6) {
        vertices[i][0] = 0;
      }
      if (um2::abs(vertices[i][1]) < 1e-6) {
        vertices[i][1] = 0;
      }
    }
    QuadraticQuadMesh<2, T, I> mesh;
    mesh.vertices = vertices;
    mesh.fv = faces;
    mesh.populateVF();
    this->quadratic_quad.push_back(um2::move(mesh));
    LOG_TRACE("Finished creating mesh");
    return mesh_id;
  }
  Log::error("Only linear and quadratic meshes are supported for a cylindrical pin mesh");
  return -1;
}

//=============================================================================
// makeRectangularPinMesh
//=============================================================================

template <std::floating_point T, std::integral I>
auto
SpatialPartition<T, I>::makeRectangularPinMesh(Vec2<T> dxdy, Size nx, Size ny)
{
  if (dxdy[0] <= 0 || dxdy[1] <= 0) {
    Log::error("Pin dimensions must be positive");
  }
  if (nx <= 0 || ny <= 0) {
    Log::error("Number of divisions in x and y must be positive");
  }

  Size const mesh_id = this->quad.size();
  Log::info("Making rectangular pin mesh " + toString(mesh_id));

  // Make the vertices
  Vector<Point2<T>> vertices((nx + 1) * (ny + 1));
  T const delta_x = dxdy[0] / nx;
  T const delta_y = dxdy[1] / ny;
  for (Size j = 0; j < ny + 1; ++j) {
    for (Size i = 0; i < nx + 1; ++i) {
      vertices[j * (nx + 1) + i] = {i * delta_x, j * delta_y};
    }
  }
  // Make the faces
  Vector<Vec<4, I>> faces(nx * ny);
  // Left to right, bottom to top
  for (Size j = 0; j < ny; ++j) {
    for (Size i = 0; i < nx; ++i) {
      faces[j * nx + i] = {(j) * (nx + 1) + i, (j) * (nx + 1) + i + 1,
                           (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i};
    }
  }
  QuadMesh<2, T, I> mesh;
  mesh.vertices = vertices;
  mesh.fv = faces;
  mesh.populateVF();
  this->quad.push_back(um2::move(mesh));
  LOG_TRACE("Finished creating mesh");
  return mesh_id;
}

//=============================================================================
// checkMeshExists
//=============================================================================

template <std::floating_point T, std::integral I>
inline void
SpatialPartition<T, I>::checkMeshExists(MeshType mesh_type, Size mesh_id) const
{
  switch (mesh_type) {
  case MeshType::Tri:
    if (0 > mesh_id || mesh_id >= this->tri.size()) {
      Log::error("Tri mesh " + toString(mesh_id) + " does not exist");
    }
    break;
  case MeshType::Quad:
    if (0 > mesh_id || mesh_id >= this->quad.size()) {
      Log::error("Quad mesh " + toString(mesh_id) + " does not exist");
    }
    break;
  case MeshType::QuadraticTri:
    if (0 > mesh_id || mesh_id >= this->quadratic_tri.size()) {
      Log::error("Quadratic tri mesh " + toString(mesh_id) + " does not exist");
    }
    break;
  case MeshType::QuadraticQuad:
    if (0 > mesh_id || mesh_id >= this->quadratic_quad.size()) {
      Log::error("Quadratic quad mesh " + toString(mesh_id) + " does not exist");
    }
    break;
  default:
    Log::error("Invalid mesh type");
  }
}

//=============================================================================
// makeCoarseCell
//=============================================================================

template <std::floating_point T, std::integral I>
auto
SpatialPartition<T, I>::makeCoarseCell(Vec2<T> const dxdy, MeshType const mesh_type,
                                       Size const mesh_id,
                                       Vector<MaterialID> const & material_ids) -> Size
{
  Size const cc_id = coarse_cells.size();
  Log::info("Making coarse cell " + toString(cc_id));
  // Ensure dx and dy are positive
  if (dxdy[0] <= 0 || dxdy[1] <= 0) {
    Log::error("dx and dy must be positive:; " + toString(dxdy[0]) + ", " +
               toString(dxdy[1]));
    return -1;
  }
  // Ensure that the mesh exists
  if (mesh_id != -1) {
    checkMeshExists(mesh_type, mesh_id);
    // Make sure materials are specified
    if (material_ids.empty()) {
      Log::error("No materials specified");
      return -1;
    }
  }

  // Create the coarse cell
  this->coarse_cells.push_back(CoarseCell{dxdy, mesh_type, mesh_id, material_ids});
  return cc_id;
}

//=============================================================================
// makeRTM
//=============================================================================

template <std::floating_point T, std::integral I>
auto
SpatialPartition<T, I>::makeRTM(Vector<Vector<Size>> const & cc_ids) -> Size
{
  Size const rtm_id = rtms.size();
  Log::info("Making ray tracing module " + toString(rtm_id));
  Vector<Size> unique_cc_ids;
  Vector<Vec2<T>> dxdy;
  // Ensure that all coarse cells exist
  Size const num_cc = coarse_cells.size();
  for (auto const & cc_ids_row : cc_ids) {
    for (auto const & id : cc_ids_row) {
      if (id < 0 || id >= num_cc) {
        Log::error("Coarse cell " + toString(id) + " does not exist");
        return -1;
      }
      auto * const it = std::find(unique_cc_ids.begin(), unique_cc_ids.end(), id);
      if (it == unique_cc_ids.end()) {
        unique_cc_ids.push_back(id);
        // We know id > 0, so subtracting 1 is safe
        dxdy.push_back(coarse_cells[id].dxdy);
      }
    }
  }
  // For a max pin ID N, the RectilinearGrid constructor needs all dxdy from 0 to N.
  // To get around this requirement, we will renumber the coarse cells to be 0, 1, 2,
  // 3, ..., and then use the renumbered IDs to create the RectilinearGrid.
  Vector<Vector<Size>> cc_ids_renumbered(cc_ids.size());
  for (Size i = 0; i < cc_ids.size(); ++i) {
    cc_ids_renumbered[i].resize(cc_ids[i].size());
    for (Size j = 0; j < cc_ids[i].size(); ++j) {
      auto * const it =
          std::find(unique_cc_ids.begin(), unique_cc_ids.end(), cc_ids[i][j]);
      ASSERT(it != unique_cc_ids.cend());
      cc_ids_renumbered[i][j] = static_cast<Size>(it - unique_cc_ids.begin());
    }
  }
  // Create the rectilinear grid
  RectilinearGrid2<T> grid(dxdy, cc_ids_renumbered);
  // Ensure the grid has the same dxdy as all other RTMs
  if (!rtms.empty()) {
    auto const eps = eps_distance<T>;
    if (um2::abs(grid.width() - rtms[0].width()) > eps ||
        um2::abs(grid.height() - rtms[0].height()) > eps) {
      Log::error("All RTMs must have the same dxdy");
      return -1;
    }
  }
  // Flatten the coarse cell IDs (rows are reversed)
  Size const num_rows = cc_ids.size();
  Size const num_cols = cc_ids[0].size();
  Vector<I> cc_ids_flat(num_rows * num_cols);
  for (Size i = 0; i < num_rows; ++i) {
    for (Size j = 0; j < num_cols; ++j) {
      cc_ids_flat[i * num_cols + j] = static_cast<I>(cc_ids[num_rows - 1 - i][j]);
    }
  }
  RTM rtm;
  rtm.grid = um2::move(grid);
  rtm.children = um2::move(cc_ids_flat);
  rtms.push_back(um2::move(rtm));
  return rtm_id;
}

//=============================================================================
// makeLattice
//=============================================================================

template <std::floating_point T, std::integral I>
auto
SpatialPartition<T, I>::stdMakeLattice(std::vector<std::vector<Size>> const & rtm_ids)
    -> Size
{
  // Convert to um2::Vector
  Vector<Vector<Size>> rtm_ids_um2(static_cast<Size>(rtm_ids.size()));
  for (size_t i = 0; i < rtm_ids.size(); ++i) {
    rtm_ids_um2[static_cast<Size>(i)].resize(static_cast<Size>(rtm_ids[i].size()));
    for (size_t j = 0; j < rtm_ids[i].size(); ++j) {
      rtm_ids_um2[static_cast<Size>(i)][static_cast<Size>(j)] =
          static_cast<Size>(rtm_ids[i][j]);
    }
  }
  return makeLattice(rtm_ids_um2);
}

template <std::floating_point T, std::integral I>
auto
SpatialPartition<T, I>::makeLattice(Vector<Vector<Size>> const & rtm_ids) -> Size
{
  Size const lat_id = lattices.size();
  Log::info("Making lattice " + toString(lat_id));
  // Ensure that all RTMs exist
  Size const num_rtm = rtms.size();
  for (auto const & rtm_ids_row : rtm_ids) {
    auto const * const it =
        std::find_if(rtm_ids_row.begin(), rtm_ids_row.end(),
                     [num_rtm](Size const id) { return id < 0 || id >= num_rtm; });
    if (it != rtm_ids_row.cend()) {
      Log::error("RTM " + toString(*it) + " does not exist");
      return -1;
    }
  }
  // Create the lattice
  // Ensure each row has the same number of columns
  Point2<T> const minima(0, 0);
  Vec2<T> const spacing(rtms[0].width(), rtms[0].height());
  Size const num_rows = rtm_ids.size();
  Size const num_cols = rtm_ids[0].size();
  for (Size i = 1; i < num_rows; ++i) {
    if (rtm_ids[i].size() != num_cols) {
      Log::error("Each row must have the same number of columns");
      return -1;
    }
  }
  Vec2<Size> const num_cells(num_cols, num_rows);
  RegularGrid2<T> grid(minima, spacing, num_cells);
  // Flatten the RTM IDs (rows are reversed)
  Vector<I> rtm_ids_flat(num_rows * num_cols);
  for (Size i = 0; i < num_rows; ++i) {
    for (Size j = 0; j < num_cols; ++j) {
      rtm_ids_flat[i * num_cols + j] = static_cast<I>(rtm_ids[num_rows - 1 - i][j]);
    }
  }
  Lattice lat;
  lat.grid = um2::move(grid);
  lat.children = um2::move(rtm_ids_flat);
  lattices.push_back(um2::move(lat));
  return lat_id;
}

//=============================================================================
// makeAssembly
//=============================================================================

template <std::floating_point T, std::integral I>
auto
SpatialPartition<T, I>::makeAssembly(Vector<Size> const & lat_ids, Vector<T> const & z)
    -> Size
{
  Size const asy_id = assemblies.size();
  Log::info("Making assembly " + toString(asy_id));
  // Ensure that all lattices exist
  Size const num_lat = lattices.size();
  {
    auto const * const it =
        std::find_if(lat_ids.cbegin(), lat_ids.cend(),
                     [num_lat](Size const id) { return id < 0 || id >= num_lat; });
    if (it != lat_ids.end()) {
      Log::error("Lattice " + toString(*it) + " does not exist");
      return -1;
    }
  }
  // Ensure the number of lattices is 1 less than the number of z-planes
  if (lat_ids.size() + 1 != z.size()) {
    Log::error("The number of lattices must be 1 less than the number of z-planes");
    return -1;
  }
  // Ensure all z-planes are in ascending order
  if (!std::is_sorted(z.begin(), z.end())) {
    Log::error("The z-planes must be in ascending order");
    return -1;
  }
  // Ensure this assembly is the same height as all other assemblies
  if (!assemblies.empty()) {
    auto const eps = eps_distance<T>;
    T const assem_top = assemblies[0].xMax();
    T const assem_bot = assemblies[0].xMin();
    if (um2::abs(z.back() - assem_top) > eps || um2::abs(z.front() - assem_bot) > eps) {
      Log::error("All assemblies must have the same height");
      return -1;
    }
  }
  // Ensure the lattices all have the same dimensions. Since they are composed of RTMs,
  // it is sufficient to check numXCells and numYCells.
  {
    Size const num_xcells = lattices[lat_ids[0]].numXCells();
    Size const num_ycells = lattices[lat_ids[0]].numYCells();
    auto const * const it = std::find_if(
        lat_ids.cbegin(), lat_ids.cend(), [num_xcells, num_ycells, this](Size const id) {
          return this->lattices[id].numXCells() != num_xcells ||
                 this->lattices[id].numYCells() != num_ycells;
        });
    if (it != lat_ids.end()) {
      Log::error("All lattices must have the same xy-dimensions");
      return -1;
    }
  }

  Vector<I> lat_ids_i(lat_ids.size());
  for (Size i = 0; i < lat_ids.size(); ++i) {
    lat_ids_i[i] = static_cast<I>(lat_ids[i]);
  }

  RectilinearGrid1<T> grid;
  grid.divs[0].resize(z.size());
  um2::copy(z.cbegin(), z.cend(), grid.divs[0].begin());
  Assembly asy;
  asy.grid = um2::move(grid);
  asy.children = um2::move(lat_ids_i);
  assemblies.push_back(um2::move(asy));
  return asy_id;
}

//=============================================================================
// makeCore
//=============================================================================

template <std::floating_point T, std::integral I>
auto
SpatialPartition<T, I>::stdMakeCore(std::vector<std::vector<Size>> const & asy_ids)
    -> Size
{
  // Convert to um2::Vector
  Vector<Vector<Size>> asy_ids_um2(static_cast<Size>(asy_ids.size()));
  for (size_t i = 0; i < asy_ids.size(); ++i) {
    asy_ids_um2[static_cast<Size>(i)].resize(static_cast<Size>(asy_ids[i].size()));
    for (size_t j = 0; j < asy_ids[i].size(); ++j) {
      asy_ids_um2[static_cast<Size>(i)][static_cast<Size>(j)] =
          static_cast<Size>(asy_ids[i][j]);
    }
  }
  return makeCore(asy_ids_um2);
}

template <std::floating_point T, std::integral I>
auto
SpatialPartition<T, I>::makeCore(Vector<Vector<Size>> const & asy_ids) -> Size
{
  Log::info("Making core");
  // Ensure that all assemblies exist
  Size const num_asy = assemblies.size();
  for (auto const & asy_ids_row : asy_ids) {
    auto const * const it =
        std::find_if(asy_ids_row.cbegin(), asy_ids_row.cend(),
                     [num_asy](Size const id) { return id < 0 || id >= num_asy; });
    if (it != asy_ids_row.end()) {
      Log::error("Assembly " + toString(*it) + " does not exist");
      return -1;
    }
  }
  Vector<Vec2<T>> dxdy(num_asy);
  for (Size i = 0; i < num_asy; ++i) {
    auto const lat_id = static_cast<Size>(assemblies[i].getChild(0));
    dxdy[i] = {lattices[lat_id].width(), lattices[lat_id].height()};
  }
  // Create the rectilinear grid
  RectilinearGrid2<T> grid(dxdy, asy_ids);
  // Flatten the assembly IDs (rows are reversed)
  Size const num_rows = asy_ids.size();
  Size const num_cols = asy_ids[0].size();
  Vector<I> asy_ids_flat(num_rows * num_cols);
  for (Size i = 0; i < num_rows; ++i) {
    if (asy_ids[i].size() != num_cols) {
      Log::error("Each row must have the same number of columns");
      return -1;
    }
    for (Size j = 0; j < num_cols; ++j) {
      asy_ids_flat[i * num_cols + j] = static_cast<I>(asy_ids[num_rows - 1 - i][j]);
    }
  }
  core.grid = um2::move(grid);
  core.children = um2::move(asy_ids_flat);
  return 0;
}

//=============================================================================
// importCoarseCells
//=============================================================================

template <std::floating_point T, std::integral I>
void
SpatialPartition<T, I>::importCoarseCells(String const & filename)
{
  Log::info("Importing coarse cells from " + filename);
  PolytopeSoup<T, I> mesh_file;
  mesh_file.read(filename);

  // Get the materials
  Vector<String> material_names;
  mesh_file.getMaterialNames(material_names);
  materials.resize(material_names.size());
  for (Size i = 0; i < material_names.size(); ++i) {
    ShortString & this_name = materials[i].name;
    this_name = ShortString(material_names[i].substr(9).c_str());
  }

  // For each coarse cell
  std::stringstream ss;
  Size const num_coarse_cells = numCoarseCells();
  for (Size i = 0; i < num_coarse_cells; ++i) {
    // Get the submesh for the coarse cell
    ss.str("");
    ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << i;
    String const cc_name(ss.str().c_str());
    PolytopeSoup<T, I> cc_submesh;
    mesh_file.getSubmesh(cc_name, cc_submesh);

    // Get the mesh type and material IDs
    MeshType const mesh_type = cc_submesh.getMeshType();
    CoarseCell & cc = coarse_cells[i];
    cc.mesh_type = mesh_type;
    Vector<MaterialID> mat_ids;
    cc_submesh.getMaterialIDs(mat_ids, material_names);
    cc.material_ids.resize(mat_ids.size());
    um2::copy(mat_ids.cbegin(), mat_ids.cend(), cc.material_ids.begin());

    // Create the FaceVertexMesh and shift it from global coordinates to local
    // coordinates, with the bottom left corner of the AABB at the origin
    AxisAlignedBox2<T> bb;
    Point2<T> * vertices = nullptr;
    Size const num_verts = cc_submesh.numVerts();
    switch (mesh_type) {
    case MeshType::Tri:
      cc.mesh_id = tri.size();
      tri.push_back(um2::move(TriMesh<2, T, I>(cc_submesh)));
      bb = tri.back().boundingBox();
      vertices = tri.back().vertices.data();
      break;
    case MeshType::Quad:
      cc.mesh_id = quad.size();
      quad.push_back(um2::move(QuadMesh<2, T, I>(cc_submesh)));
      bb = quad.back().boundingBox();
      vertices = quad.back().vertices.data();
      break;
    case MeshType::QuadraticTri:
      cc.mesh_id = quadratic_tri.size();
      quadratic_tri.push_back(um2::move(QuadraticTriMesh<2, T, I>(cc_submesh)));
      bb = quadratic_tri.back().boundingBox();
      vertices = quadratic_tri.back().vertices.data();
      break;
    case MeshType::QuadraticQuad:
      cc.mesh_id = quadratic_quad.size();
      quadratic_quad.push_back(um2::move(QuadraticQuadMesh<2, T, I>(cc_submesh)));
      bb = quadratic_quad.back().boundingBox();
      vertices = quadratic_quad.back().vertices.data();
      break;
    default:
      Log::error("Mesh type not supported");
    }

    // Shift the points so that the min point is at the origin.
    Point2<T> const min_point = bb.minima;
    for (Size ip = 0; ip < num_verts; ++ip) {
      vertices[ip] -= min_point;
    }
#ifndef NDEBUG
    Point2<T> const dxdy = bb.maxima - bb.minima;
    ASSERT(isApprox(dxdy, cc.dxdy));
#endif
  }
}

//=============================================================================
// toPolytopeSoup
//=============================================================================

template <std::floating_point T, std::integral I>
void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
SpatialPartition<T, I>::toPolytopeSoup(PolytopeSoup<T, I> & soup, bool write_kn) const
{
  LOG_DEBUG("Converting spatial partition to polytope soup");

  if (core.children.empty()) {
    Log::error("Core has no children");
    return;
  }
  // Allocate counters for each assembly, lattice, etc.
  Vector<I> asy_found(assemblies.size(), -1);
  Vector<I> lat_found(lattices.size(), -1);
  Vector<I> rtm_found(rtms.size(), -1);
  Vector<I> cc_found(coarse_cells.size(), -1);

  std::stringstream ss;
  Size total_num_faces = 0;
  LOG_DEBUG("materials.size() = " + toString(materials.size()));
  Vector<Vector<I>> material_elsets(materials.size());
  Vector<T> kn_max;
  Vector<T> kn_mean;
  Vector<Vector<T>> cc_kns_max(coarse_cells.size());
  Vector<Vector<T>> cc_kns_mean(coarse_cells.size());

  // We will encode the M by N dimensions of each assembly, lattice,
  // etc. as elset data.
  // For each assembly
  Size const nyasy = core.numYCells();
  Size const nxasy = core.numXCells();
  for (Size iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Size ixasy = 0; ixasy < nxasy; ++ixasy) {
      Size const asy_faces_prev = total_num_faces;
      auto const asy_id = static_cast<Size>(core.getChild(ixasy, iyasy));
      I const asy_id_ctr = ++asy_found[asy_id];
      // Get elset name
      ss.str("");
      ss << "Assembly_" << std::setw(5) << std::setfill('0') << asy_id << "_"
         << std::setw(5) << std::setfill('0') << asy_id_ctr;
      String const asy_name(ss.str().c_str());
      LOG_DEBUG("Assembly name: " + asy_name);
      // Get the assembly offset (lower left corner)
      AxisAlignedBox2<T> const asy_bb = core.getBox(ixasy, iyasy);
      Point2<T> const asy_ll = asy_bb.minima; // Lower left corner

      auto const & assembly = assemblies[asy_id];
      if (assembly.children.empty()) {
        Log::error("Assembly has no children");
        return;
      }

      // For each lattice
      Size const nzlat = assembly.numXCells();
      for (Size izlat = 0; izlat < nzlat; ++izlat) {
        Size const lat_faces_prev = total_num_faces;
        auto const lat_id = static_cast<Size>(assembly.getChild(izlat));
        I const lat_id_ctr = ++lat_found[lat_id];
        // Get elset name
        ss.str("");
        ss << "Lattice_" << std::setw(5) << std::setfill('0') << lat_id << "_"
           << std::setw(5) << std::setfill('0') << lat_id_ctr;
        String const lat_name(ss.str().c_str());
        LOG_DEBUG("Lattice name: " + lat_name);
        // Get the lattice offset (z direction)
        // The midplane is the location that the geometry was sampled at.
        T const low_z = assembly.grid.divs[0][izlat];
        T const high_z = assembly.grid.divs[0][izlat + 1];
        T const lat_z = (low_z + high_z) / 2;

        // Get the lattice
        auto const & lattice = lattices[lat_id];
        if (lattice.children.empty()) {
          Log::error("Lattice has no children");
          return;
        }

        // For each RTM
        Size const nyrtm = lattice.numYCells();
        Size const nxrtm = lattice.numXCells();
        for (Size iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (Size ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {
            Size const rtm_faces_prev = total_num_faces;
            auto const rtm_id = static_cast<Size>(lattice.getChild(ixrtm, iyrtm));
            I const rtm_id_ctr = ++rtm_found[rtm_id];
            ss.str("");
            ss << "RTM_" << std::setw(5) << std::setfill('0') << rtm_id << "_"
               << std::setw(5) << std::setfill('0') << rtm_id_ctr;
            String const rtm_name(ss.str().c_str());
            LOG_DEBUG("RTM name: " + rtm_name);
            // Get the RTM offset (lower left corner)
            auto const rtm_bb = lattice.getBox(ixrtm, iyrtm);
            Point2<T> const rtm_ll = rtm_bb.minima; // Lower left corner

            // Get the rtm
            auto const & rtm = rtms[rtm_id];
            if (rtm.children.empty()) {
              Log::error("RTM has no children");
              return;
            }

            Size const nycells = rtm.numYCells();
            Size const nxcells = rtm.numXCells();
            for (Size iycell = 0; iycell < nycells; ++iycell) {
              for (Size ixcell = 0; ixcell < nxcells; ++ixcell) {
                Size const cell_faces_prev = total_num_faces;
                auto const & cell_id = static_cast<Size>(rtm.getChild(ixcell, iycell));
                I const cell_id_ctr = ++cc_found[cell_id];
                ss.str("");
                ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << cell_id
                   << "_" << std::setw(5) << std::setfill('0') << cell_id_ctr;
                String const cell_name(ss.str().c_str());
                LOG_DEBUG("Coarse cell name: " + cell_name);
                // Get the cell offset (lower left corner)
                auto const cell_bb = rtm.getBox(ixcell, iycell);
                Point2<T> const cell_ll = cell_bb.minima; // Lower left corner

                // Get the mesh type and id of the coarse cell.
                MeshType const mesh_type = coarse_cells[cell_id].mesh_type;
                Size const mesh_id = coarse_cells[cell_id].mesh_id;
                LOG_TRACE("mesh_id = " + toString(mesh_id));
                // Add to material elsets
                Vector<MaterialID> const & cell_materials =
                    coarse_cells[cell_id].material_ids;
                LOG_TRACE("cell_materials.size() = " + toString(cell_materials.size()));
                for (Size iface = 0; iface < cell_materials.size(); ++iface) {
                  auto const mat_id = static_cast<Size>(
                      static_cast<uint32_t>(cell_materials[iface]));
                  material_elsets[mat_id].push_back(
                      static_cast<I>(iface + cell_faces_prev));
                }

                Point2<T> const * fvm_vertices_begin = nullptr;
                Point2<T> const * fvm_vertices_end = nullptr;
                switch (mesh_type) {
                case MeshType::Tri:
                  LOG_TRACE("Mesh type: Tri");
                  fvm_vertices_begin = tri[mesh_id].vertices.begin();
                  fvm_vertices_end = tri[mesh_id].vertices.end();
                  break;
                case MeshType::Quad:
                  LOG_TRACE("Mesh type: Quad");
                  fvm_vertices_begin = quad[mesh_id].vertices.begin();
                  fvm_vertices_end = quad[mesh_id].vertices.end();
                  break;
                case MeshType::QuadraticTri:
                  LOG_TRACE("Mesh type: QuadraticTri");
                  fvm_vertices_begin = quadratic_tri[mesh_id].vertices.begin();
                  fvm_vertices_end = quadratic_tri[mesh_id].vertices.end();
                  break;
                case MeshType::QuadraticQuad:
                  LOG_TRACE("Mesh type: QuadraticQuad");
                  fvm_vertices_begin = quadratic_quad[mesh_id].vertices.begin();
                  fvm_vertices_end = quadratic_quad[mesh_id].vertices.end();
                  break;
                default:
                  Log::error("Unsupported mesh type");
                  return;
                } // switch (mesh_type)

                // Add each vertex to the PolytopeSoup, offsetting by the
                // global xyz offset
                auto const num_verts_prev = static_cast<I>(soup.numVerts());
                Point2<T> const xy_offset = cell_ll + rtm_ll + asy_ll;
                for (auto it = fvm_vertices_begin; it != fvm_vertices_end; ++it) {
                  Point2<T> const p = *it + xy_offset;
                  soup.addVertex(p[0], p[1], lat_z);
                }

                // Add each face to the PolytopeSoup, offsetting by num_verts_prev
                LOG_TRACE("Adding faces to PolytopeSoup");
                switch (mesh_type) {
                case MeshType::Tri: {
                  Size const verts_per_face = 3;
                  VTKElemType const elem_type = VTKElemType::Triangle;
                  Vector<I> conn(verts_per_face);
                  LOG_TRACE("tri[mesh_id].fv.size() = " +
                            toString(tri[mesh_id].fv.size()));
                  for (Size iface = 0; iface < tri[mesh_id].fv.size(); ++iface) {
                    auto const & face_conn = tri[mesh_id].fv[iface];
                    for (Size i = 0; i < verts_per_face; ++i) {
                      conn[i] = face_conn[i] + num_verts_prev;
                    }
                    soup.addElement(elem_type, conn);
                  }
                  if (write_kn) {
                    if (cc_kns_max[cell_id].empty()) {
                      LOG_TRACE("Computing Knudsen numbers");
                      for (Size iface = 0; iface < tri[mesh_id].fv.size(); ++iface) {
                        T const mcl = tri[mesh_id].getFace(iface).meanChordLength();
                        auto const mat_id = static_cast<Size>(
                            static_cast<uint32_t>(cell_materials[iface]));
                        T const t_max = materials[mat_id].xs.getOneGroupTotalXS(
                            XSReductionStrategy::Max);
                        T const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
                            XSReductionStrategy::Mean);
                        cc_kns_max[cell_id].push_back(static_cast<T>(1) / (t_max * mcl));
                        cc_kns_mean[cell_id].push_back(static_cast<T>(1) /
                                                       (t_mean * mcl));
                      }
                    }
                    for (auto const & kn : cc_kns_max[cell_id]) {
                      // cppcheck-suppress useStlAlgorithm
                      kn_max.push_back(kn);
                    }
                    for (auto const & kn : cc_kns_mean[cell_id]) {
                      // cppcheck-suppress useStlAlgorithm
                      kn_mean.push_back(kn);
                    }
                  }
                } break;
                case MeshType::Quad: {
                  Size const verts_per_face = 4;
                  VTKElemType const elem_type = VTKElemType::Quad;
                  Vector<I> conn(verts_per_face);
                  for (Size iface = 0; iface < quad[mesh_id].fv.size(); ++iface) {
                    auto const & face_conn = quad[mesh_id].fv[iface];
                    for (Size i = 0; i < verts_per_face; ++i) {
                      conn[i] = face_conn[i] + num_verts_prev;
                    }
                    soup.addElement(elem_type, conn);
                  }
                  if (write_kn) {
                    if (cc_kns_max[cell_id].empty()) {
                      LOG_TRACE("Computing Knudsen numbers");
                      for (Size iface = 0; iface < quad[mesh_id].fv.size(); ++iface) {
                        LOG_TRACE("face = " + toString(quad[mesh_id].fv[iface][0]) + " " +
                                  toString(quad[mesh_id].fv[iface][1]) + " " +
                                  toString(quad[mesh_id].fv[iface][2]) + " " +
                                  toString(quad[mesh_id].fv[iface][3]));
                        T const mcl = quad[mesh_id].getFace(iface).meanChordLength();
                        auto const mat_id = static_cast<Size>(
                            static_cast<uint32_t>(cell_materials[iface]));
                        T const t_max = materials[mat_id].xs.getOneGroupTotalXS(
                            XSReductionStrategy::Max);
                        T const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
                            XSReductionStrategy::Mean);
                        cc_kns_max[cell_id].push_back(static_cast<T>(1) / (t_max * mcl));
                        cc_kns_mean[cell_id].push_back(static_cast<T>(1) /
                                                       (t_mean * mcl));
                      }
                    }
                    for (auto const & kn : cc_kns_max[cell_id]) {
                      // cppcheck-suppress useStlAlgorithm
                      kn_max.push_back(kn);
                    }
                    for (auto const & kn : cc_kns_mean[cell_id]) {
                      // cppcheck-suppress useStlAlgorithm
                      kn_mean.push_back(kn);
                    }
                  }
                } break;
                case MeshType::QuadraticTri: {
                  Size const verts_per_face = 6;
                  VTKElemType const elem_type = VTKElemType::QuadraticTriangle;
                  Vector<I> conn(verts_per_face);
                  for (Size iface = 0; iface < quadratic_tri[mesh_id].fv.size();
                       ++iface) {
                    auto const & face_conn = quadratic_tri[mesh_id].fv[iface];
                    for (Size i = 0; i < verts_per_face; ++i) {
                      conn[i] = face_conn[i] + num_verts_prev;
                    }
                    soup.addElement(elem_type, conn);
                  }
                  if (write_kn) {
                    if (cc_kns_max[cell_id].empty()) {
                      LOG_TRACE("Computing Knudsen numbers");
                      for (Size iface = 0; iface < quadratic_tri[mesh_id].fv.size();
                           ++iface) {
                        T const mcl =
                            quadratic_tri[mesh_id].getFace(iface).meanChordLength();
                        auto const mat_id = static_cast<Size>(
                            static_cast<uint32_t>(cell_materials[iface]));
                        T const t_max = materials[mat_id].xs.getOneGroupTotalXS(
                            XSReductionStrategy::Max);
                        T const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
                            XSReductionStrategy::Mean);
                        cc_kns_max[cell_id].push_back(static_cast<T>(1) / (t_max * mcl));
                        cc_kns_mean[cell_id].push_back(static_cast<T>(1) /
                                                       (t_mean * mcl));
                      }
                    }
                    for (auto const & kn : cc_kns_max[cell_id]) {
                      // cppcheck-suppress useStlAlgorithm
                      kn_max.push_back(kn);
                    }
                    for (auto const & kn : cc_kns_mean[cell_id]) {
                      // cppcheck-suppress useStlAlgorithm
                      kn_mean.push_back(kn);
                    }
                  }
                } break;
                case MeshType::QuadraticQuad: {
                  Size const verts_per_face = 8;
                  VTKElemType const elem_type = VTKElemType::QuadraticQuad;
                  Vector<I> conn(verts_per_face);
                  for (Size iface = 0; iface < quadratic_quad[mesh_id].fv.size();
                       ++iface) {
                    auto const & face_conn = quadratic_quad[mesh_id].fv[iface];
                    for (Size i = 0; i < verts_per_face; ++i) {
                      conn[i] = face_conn[i] + num_verts_prev;
                    }
                    soup.addElement(elem_type, conn);
                  }
                  if (write_kn) {
                    if (cc_kns_max[cell_id].empty()) {
                      LOG_TRACE("Computing Knudsen numbers");
                      for (Size iface = 0; iface < quadratic_quad[mesh_id].fv.size();
                           ++iface) {
                        T const mcl =
                            quadratic_quad[mesh_id].getFace(iface).meanChordLength();
                        auto const mat_id = static_cast<Size>(
                            static_cast<uint32_t>(cell_materials[iface]));
                        T const t_max = materials[mat_id].xs.getOneGroupTotalXS(
                            XSReductionStrategy::Max);
                        T const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
                            XSReductionStrategy::Mean);
                        cc_kns_max[cell_id].push_back(static_cast<T>(1) / (t_max * mcl));
                        cc_kns_mean[cell_id].push_back(static_cast<T>(1) /
                                                       (t_mean * mcl));
                      }
                    }
                    for (auto const & kn : cc_kns_max[cell_id]) {
                      // cppcheck-suppress useStlAlgorithm
                      kn_max.push_back(kn);
                    }
                    for (auto const & kn : cc_kns_mean[cell_id]) {
                      // cppcheck-suppress useStlAlgorithm
                      kn_mean.push_back(kn);
                    }
                  }
                } break;
                default:
                  Log::error("Unsupported mesh type");
                  return;
                } // switch (mesh_type)
                Size const num_faces = soup.numElems() - cell_faces_prev;

                // Add an elset for the cell
                Vector<I> cell_ids(num_faces);
                um2::iota(cell_ids.begin(), cell_ids.end(),
                          static_cast<I>(cell_faces_prev));
                soup.addElset(cell_name, cell_ids);
                total_num_faces += num_faces;

              } // for (ixcell)
            }   // for (iycell)

            // Add the RTM elset
            Vector<I> rtm_ids(total_num_faces - rtm_faces_prev);
            um2::iota(rtm_ids.begin(), rtm_ids.end(), static_cast<I>(rtm_faces_prev));
            soup.addElset(rtm_name, rtm_ids);
          } // for (ixrtm)
        }   // for (iyrtm)

        // Add the lattice elset
        Vector<I> lat_ids(total_num_faces - lat_faces_prev);
        um2::iota(lat_ids.begin(), lat_ids.end(), static_cast<I>(lat_faces_prev));
        soup.addElset(lat_name, lat_ids);
      } // for (izlat)

      // Add the assembly elset
      Vector<I> asy_ids(total_num_faces - asy_faces_prev);
      um2::iota(asy_ids.begin(), asy_ids.end(), static_cast<I>(asy_faces_prev));
      soup.addElset(asy_name, asy_ids);
    } // for (ixasy)
  }   // for (iyasy)

  // Add the material elsets
  for (Size imat = 0; imat < materials.size(); ++imat) {
    String const mat_name = "Material_" + String(materials[imat].name.data());
    soup.addElset(mat_name, material_elsets[imat]);
  }

  Vector<I> all_ids(total_num_faces);
  um2::iota(all_ids.begin(), all_ids.end(), static_cast<I>(0));
  // Add the knudsen number elsets
  if (write_kn) {
    soup.addElset("Knudsen_Max", all_ids, kn_max);
    soup.addElset("Knudsen_Mean", all_ids, kn_mean);
  }

  soup.sortElsets();
} // toPolytopeSoup

//==============================================================================
// getMaterialNames
//==============================================================================

template <std::floating_point T, std::integral I>
void
SpatialPartition<T, I>::getMaterialNames(Vector<String> & material_names) const
{
  material_names.clear();
  String const mat_prefix = "Material_";
  for (auto const & material : materials) {
    String const mat_suffix(material.name.data());
    material_names.push_back(mat_prefix + mat_suffix);
  }
  std::sort(material_names.begin(), material_names.end());
} // getMaterialNames

//==============================================================================
// writeXDMF
//==============================================================================

template <std::floating_point T, std::integral I>
void
// NOLINTNEXTLINE
SpatialPartition<T, I>::writeXDMF(String const & filepath, bool write_kn) const
{
  Log::info("Writing XDMF file: " + filepath);

  // Setup HDF5 file
  // Get the h5 file name
  Size last_slash = filepath.find_last_of('/');
  if (last_slash == String::npos) {
    last_slash = 0;
  }
  Size const h5filepath_end = last_slash == 0 ? 0 : last_slash + 1;
  LOG_DEBUG("h5filepath_end: " + toString(h5filepath_end));
  String const h5filename =
      filepath.substr(h5filepath_end, filepath.size() - 5 - h5filepath_end) + ".h5";
  LOG_DEBUG("h5filename: " + h5filename);
  String const h5filepath = filepath.substr(0, h5filepath_end);
  LOG_DEBUG("h5filepath: " + h5filepath);
  H5::H5File h5file((h5filepath + h5filename).c_str(), H5F_ACC_TRUNC);

  // Setup XML file
  pugi::xml_document xdoc;

  // XDMF root node
  pugi::xml_node xroot = xdoc.append_child("Xdmf");
  xroot.append_attribute("Version") = "3.0";

  // Domain node
  pugi::xml_node xdomain = xroot.append_child("Domain");

  // Get the material names from elset names, in alphabetical order.
  Vector<String> material_names;
  getMaterialNames(material_names);
  std::sort(material_names.begin(), material_names.end());

  // If there are any materials, add an information node listing them
  if (!material_names.empty()) {
    pugi::xml_node xinfo = xdomain.append_child("Information");
    xinfo.append_attribute("Name") = "Materials";
    String mats;
    for (Size i = 0; i < material_names.size(); ++i) {
      auto const & mat_name = material_names[i];
      String const short_name = mat_name.substr(9, mat_name.size() - 9);
      mats += short_name;
      if (i + 1 < material_names.size()) {
        mats += ", ";
      }
    }
    xinfo.append_child(pugi::node_pcdata).set_value(mats.c_str());
  }

  String const name = h5filename.substr(0, h5filename.size() - 3);

  // Core grid
  pugi::xml_node xcore_grid = xdomain.append_child("Grid");
  xcore_grid.append_attribute("Name") = name.c_str();
  xcore_grid.append_attribute("GridType") = "Tree";

  // h5
  H5::Group const h5core_group = h5file.createGroup(name.c_str());
  String const h5core_grouppath = "/" + name;

  // Allocate counters for each assembly, lattice, etc.
  Vector<I> asy_found(assemblies.size(), -1);
  Vector<I> lat_found(lattices.size(), -1);
  Vector<I> rtm_found(rtms.size(), -1);
  Vector<I> cc_found(coarse_cells.size(), -1);

  std::stringstream ss;
  Vector<PolytopeSoup<T, I>> soups(coarse_cells.size());
  Vector<Vector<T>> cc_kns_max(coarse_cells.size());
  Vector<Vector<T>> cc_kns_mean(coarse_cells.size());

  if (core.children.empty()) {
    Log::error("Core has no children");
    return;
  }
  Size const nyasy = core.numYCells();
  Size const nxasy = core.numXCells();
  // Core M by N
  pugi::xml_node xcore_info = xcore_grid.append_child("Information");
  xcore_info.append_attribute("Name") = "M_by_N";
  String const core_mn_str = toString(nyasy) + " x " + toString(nxasy);
  xcore_info.append_child(pugi::node_pcdata).set_value(core_mn_str.c_str());
  // For each assembly
  for (Size iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Size ixasy = 0; ixasy < nxasy; ++ixasy) {
      auto const asy_id = static_cast<Size>(core.getChild(ixasy, iyasy));
      I const asy_id_ctr = ++asy_found[asy_id];
      // Get elset name
      ss.str("");
      ss << "Assembly_" << std::setw(5) << std::setfill('0') << asy_id << "_"
         << std::setw(5) << std::setfill('0') << asy_id_ctr;
      String const asy_name(ss.str().c_str());
      LOG_DEBUG("Assembly name: " + asy_name);
      // Get the assembly offset (lower left corner)
      AxisAlignedBox2<T> const asy_bb = core.getBox(ixasy, iyasy);
      Point2<T> const asy_ll = asy_bb.minima; // Lower left corner

      auto const & assembly = assemblies[asy_id];
      if (assembly.children.empty()) {
        Log::error("Assembly has no children");
        return;
      }

      // Create the XML grid
      pugi::xml_node xasy_grid = xcore_grid.append_child("Grid");
      xasy_grid.append_attribute("Name") = ss.str().c_str();
      xasy_grid.append_attribute("GridType") = "Tree";

      // Write the M by N information
      Size const nzlat = assembly.numXCells();
      pugi::xml_node xasy_info = xasy_grid.append_child("Information");
      xasy_info.append_attribute("Name") = "M_by_N";
      String const asy_mn_str = toString(nzlat) + " x 1";
      xasy_info.append_child(pugi::node_pcdata).set_value(asy_mn_str.c_str());

      // Create the h5 group
      String const h5asy_grouppath = h5core_grouppath + "/" + String(ss.str().c_str());
      H5::Group const h5asy_group = h5file.createGroup(h5asy_grouppath.c_str());

      // For each lattice
      for (Size izlat = 0; izlat < nzlat; ++izlat) {
        auto const lat_id = static_cast<Size>(assembly.getChild(izlat));
        I const lat_id_ctr = ++lat_found[lat_id];
        // Get elset name
        ss.str("");
        ss << "Lattice_" << std::setw(5) << std::setfill('0') << lat_id << "_"
           << std::setw(5) << std::setfill('0') << lat_id_ctr;
        String const lat_name(ss.str().c_str());
        LOG_DEBUG("Lattice name: " + lat_name);
        // Get the lattice offset (z direction)
        // The midplane is the location that the geometry was sampled at.
        T const low_z = assembly.grid.divs[0][izlat];
        T const high_z = assembly.grid.divs[0][izlat + 1];
        T const lat_z = (low_z + high_z) / 2;

        // Get the lattice
        auto const & lattice = lattices[lat_id];
        if (lattice.children.empty()) {
          Log::error("Lattice has no children");
          return;
        }

        // Create the XML grid
        pugi::xml_node xlat_grid = xasy_grid.append_child("Grid");
        xlat_grid.append_attribute("Name") = ss.str().c_str();
        xlat_grid.append_attribute("GridType") = "Tree";

        // Add the Z information for the lattice
        pugi::xml_node xlat_info = xlat_grid.append_child("Information");
        xlat_info.append_attribute("Name") = "Z";
        String const z_values = toString(low_z) + ", " + toString(lat_z) + ", " + toString(high_z);
        xlat_info.append_child(pugi::node_pcdata).set_value(z_values.c_str());

        // Write the M by N information
        Size const nyrtm = lattice.numYCells();
        Size const nxrtm = lattice.numXCells();
        pugi::xml_node xlat_mn_info = xlat_grid.append_child("Information");
        xlat_mn_info.append_attribute("Name") = "M_by_N";
        String const lat_mn_str = toString(nyrtm) + " x " + toString(nxrtm);
        xlat_mn_info.append_child(pugi::node_pcdata).set_value(lat_mn_str.c_str());

        // Create the h5 group
        String const h5lat_grouppath = h5asy_grouppath + "/" + String(ss.str().c_str());
        H5::Group const h5lat_group = h5file.createGroup(h5lat_grouppath.c_str());

        // For each RTM
        for (Size iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (Size ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {
            auto const rtm_id = static_cast<Size>(lattice.getChild(ixrtm, iyrtm));
            I const rtm_id_ctr = ++rtm_found[rtm_id];
            ss.str("");
            ss << "RTM_" << std::setw(5) << std::setfill('0') << rtm_id << "_"
               << std::setw(5) << std::setfill('0') << rtm_id_ctr;
            String const rtm_name(ss.str().c_str());
            LOG_DEBUG("RTM name: " + rtm_name);
            // Get the RTM offset (lower left corner)
            auto const rtm_bb = lattice.getBox(ixrtm, iyrtm);
            Point2<T> const rtm_ll = rtm_bb.minima; // Lower left corner

            // Get the rtm
            auto const & rtm = rtms[rtm_id];
            if (rtm.children.empty()) {
              Log::error("RTM has no children");
              return;
            }

            // Create the XML grid
            pugi::xml_node xrtm_grid = xlat_grid.append_child("Grid");
            xrtm_grid.append_attribute("Name") = ss.str().c_str();
            xrtm_grid.append_attribute("GridType") = "Tree";

            // Write the M by N information
            Size const nycells = rtm.numYCells();
            Size const nxcells = rtm.numXCells();
            pugi::xml_node xrtm_mn_info = xrtm_grid.append_child("Information");
            xrtm_mn_info.append_attribute("Name") = "M_by_N";
            String const rtm_mn_str = toString(nycells) + " x " + toString(nxcells);
            xrtm_mn_info.append_child(pugi::node_pcdata).set_value(rtm_mn_str.c_str());

            // Create the h5 group
            String const h5rtm_grouppath = h5lat_grouppath + "/" + String(ss.str().c_str());
            H5::Group const h5rtm_group = h5file.createGroup(h5rtm_grouppath.c_str());

            for (Size iycell = 0; iycell < nycells; ++iycell) {
              for (Size ixcell = 0; ixcell < nxcells; ++ixcell) {
                auto const & cell_id = static_cast<Size>(rtm.getChild(ixcell, iycell));
                I const cell_id_ctr = ++cc_found[cell_id];
                ss.str("");
                ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << cell_id
                   << "_" << std::setw(5) << std::setfill('0') << cell_id_ctr;
                String const cell_name(ss.str().c_str());
                LOG_DEBUG("Coarse cell name: " + cell_name);
                // Get the cell offset (lower left corner)
                auto const cell_bb = rtm.getBox(ixcell, iycell);
                Point2<T> const cell_ll = cell_bb.minima; // Lower left corner

                // Get the mesh type and id of the coarse cell.
                MeshType const mesh_type = coarse_cells[cell_id].mesh_type;
                Size const mesh_id = coarse_cells[cell_id].mesh_id;
                LOG_TRACE("mesh_id = " + toString(mesh_id));
                // Add to material elsets
                Vector<MaterialID> const & cell_materials =
                    coarse_cells[cell_id].material_ids;
                LOG_TRACE("cell_materials.size() = " + toString(cell_materials.size()));

                // Convert the mesh into PolytopeSoup
                PolytopeSoup<T, I> & soup = soups[cell_id];
                if (soups[cell_id].numElems() == 0) {
                  switch (mesh_type) {
                  case MeshType::Tri:
                    LOG_TRACE("Mesh type: Tri");
                    tri[mesh_id].toPolytopeSoup(soup);
                    if (write_kn) {
                      if (cc_kns_max[cell_id].empty()) {
                        LOG_TRACE("Computing Knudsen numbers");
                        cc_kns_max[cell_id].resize(tri[mesh_id].fv.size());
                        cc_kns_mean[cell_id].resize(tri[mesh_id].fv.size());
                        for (Size iface = 0; iface < tri[mesh_id].fv.size(); ++iface) {
                          T const mcl = tri[mesh_id].getFace(iface).meanChordLength();
                          auto const mat_id = static_cast<Size>(
                              static_cast<uint32_t>(cell_materials[iface]));
                          T const t_max = materials[mat_id].xs.getOneGroupTotalXS(
                              XSReductionStrategy::Max);
                          T const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
                              XSReductionStrategy::Mean);
                          cc_kns_max[cell_id][iface] = static_cast<T>(1) / (t_max * mcl);
                          cc_kns_mean[cell_id][iface] = static_cast<T>(1) / (t_mean * mcl);
                        }
                      }
                    }
                    break;
                  case MeshType::Quad:
                    LOG_TRACE("Mesh type: Quad");
                    quad[mesh_id].toPolytopeSoup(soup);
                    if (write_kn) {
                      if (cc_kns_max[cell_id].empty()) {
                        LOG_TRACE("Computing Knudsen numbers");
                        cc_kns_max[cell_id].resize(quad[mesh_id].fv.size());
                        cc_kns_mean[cell_id].resize(quad[mesh_id].fv.size());
                        for (Size iface = 0; iface < quad[mesh_id].fv.size(); ++iface) {
                          T const mcl = quad[mesh_id].getFace(iface).meanChordLength();
                          auto const mat_id = static_cast<Size>(
                              static_cast<uint32_t>(cell_materials[iface]));
                          T const t_max = materials[mat_id].xs.getOneGroupTotalXS(
                              XSReductionStrategy::Max);
                          T const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
                              XSReductionStrategy::Mean);
                          cc_kns_max[cell_id][iface] = static_cast<T>(1) / (t_max * mcl);
                          cc_kns_mean[cell_id][iface] = static_cast<T>(1) / (t_mean * mcl);
                        }
                      }
                    }
                    break;
                  case MeshType::QuadraticTri:
                    LOG_TRACE("Mesh type: QuadraticTri");
                    quadratic_tri[mesh_id].toPolytopeSoup(soup);
                    if (write_kn) {
                      if (cc_kns_max[cell_id].empty()) {
                        LOG_TRACE("Computing Knudsen numbers");
                        cc_kns_max[cell_id].resize(quadratic_tri[mesh_id].fv.size());
                        cc_kns_mean[cell_id].resize(quadratic_tri[mesh_id].fv.size());
                        for (Size iface = 0; iface < quadratic_tri[mesh_id].fv.size(); ++iface) {
                          T const mcl = quadratic_tri[mesh_id].getFace(iface).meanChordLength();
                          auto const mat_id = static_cast<Size>(
                              static_cast<uint32_t>(cell_materials[iface]));
                          T const t_max = materials[mat_id].xs.getOneGroupTotalXS(
                              XSReductionStrategy::Max);
                          T const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
                              XSReductionStrategy::Mean);
                          cc_kns_max[cell_id][iface] = static_cast<T>(1) / (t_max * mcl);
                          cc_kns_mean[cell_id][iface] = static_cast<T>(1) / (t_mean * mcl);
                        }
                      }
                    }
                    break;
                  case MeshType::QuadraticQuad:
                    LOG_TRACE("Mesh type: QuadraticQuad");
                    quadratic_quad[mesh_id].toPolytopeSoup(soup);
                    if (write_kn) {
                      if (cc_kns_max[cell_id].empty()) {
                        LOG_TRACE("Computing Knudsen numbers");
                        cc_kns_max[cell_id].resize(quadratic_quad[mesh_id].fv.size());
                        cc_kns_mean[cell_id].resize(quadratic_quad[mesh_id].fv.size());
                        for (Size iface = 0; iface < quadratic_quad[mesh_id].fv.size(); ++iface) {
                          T const mcl = quadratic_quad[mesh_id].getFace(iface).meanChordLength();
                          auto const mat_id = static_cast<Size>(
                              static_cast<uint32_t>(cell_materials[iface]));
                          T const t_max = materials[mat_id].xs.getOneGroupTotalXS(
                              XSReductionStrategy::Max);
                          T const t_mean = materials[mat_id].xs.getOneGroupTotalXS(
                              XSReductionStrategy::Mean);
                          cc_kns_max[cell_id][iface] = static_cast<T>(1) / (t_max * mcl);
                          cc_kns_mean[cell_id][iface] = static_cast<T>(1) / (t_mean * mcl);
                        }
                      }
                    }
                    break;
                  default:
                    Log::error("Unsupported mesh type");
                    return;
                  } // switch (mesh_type)

                  // add Material elsets
                  Size const cc_nfaces = cell_materials.size();
                  Vector<I> cc_mats(cc_nfaces);
                  for (Size i = 0; i < cc_nfaces; ++i) {
                    cc_mats[i] = static_cast<I>(static_cast<uint32_t>(cell_materials[i]));
                  }
                  // Get the unique material ids
                  Vector<I> cc_mats_sorted = cc_mats;
                  std::sort(cc_mats_sorted.begin(), cc_mats_sorted.end());
                  auto * it = std::unique(cc_mats_sorted.begin(), cc_mats_sorted.end());
                  Size const cc_nunique = static_cast<Size>(it - cc_mats_sorted.begin());
                  Vector<I> cc_mats_unique(cc_nunique);
                  for (Size i = 0; i < cc_nunique; ++i) {
                    cc_mats_unique[i] = cc_mats_sorted[i];
                  }
                  // Create a vector with the face ids for each material
                  Vector<Vector<I>> cc_mats_split(cc_nunique);
                  for (Size i = 0; i < cc_nfaces; ++i) {
                    I const mat_id = cc_mats[i];
                    auto * mat_it = std::find(cc_mats_unique.begin(), cc_mats_unique.end(), mat_id);
                    Size const mat_idx = static_cast<Size>(mat_it - cc_mats_unique.begin());
                    cc_mats_split[mat_idx].push_back(i);
                  }
                  // add each material elset
                  for (Size i = 0; i < cc_nunique; ++i) {
                    I const mat_id = cc_mats_unique[i];
                    Vector<I> const & mat_faces = cc_mats_split[i];
                    String const mat_name = "Material_" + String(materials[mat_id].name.data());
                    soup.addElset(mat_name, mat_faces);
                  }

                  if (write_kn) {
                    Vector<I> all_faces(cc_nfaces);
                    um2::iota(all_faces.begin(), all_faces.end(), 0);
                    soup.addElset("Knudsen_Max", all_faces, cc_kns_max[cell_id]);
                    soup.addElset("Knudsen_Mean", all_faces, cc_kns_mean[cell_id]);
                    Vector<T> kns_max = cc_kns_max[cell_id];
                    Vector<T> kns_mean = cc_kns_mean[cell_id];
                    std::sort(kns_max.begin(), kns_max.end());
                    std::sort(kns_mean.begin(), kns_mean.end());
                    T const kn_max_max = kns_max.back();
                    T const kn_mean_max = kns_mean.back();
                    T const kn_max_min = kns_max.front();
                    T const kn_mean_min = kns_mean.front();
                    T const kn_max_mean = um2::mean(kns_max.begin(), kns_max.end());
                    T const kn_mean_mean = um2::mean(kns_mean.begin(), kns_mean.end());
                    LOG_INFO("Coarse Cell " + toString(cell_id) + " " + toString(kn_max_max) + " " + toString(kn_max_min) + " " + toString(kn_max_mean));
                    LOG_INFO("Coarse Cell " + toString(cell_id) + " " + toString(kn_mean_max) + " " + toString(kn_mean_min) + " " + toString(kn_mean_mean));
                  }

                }
                
                // Shift the mesh to global coordinates
                Point2<T> const xy_offset = cell_ll + rtm_ll + asy_ll;
                Point3<T> const shift = Point3<T>(xy_offset[0], xy_offset[1], lat_z);
                soup.translate(shift);

                // Write the mesh
                soup.writeXDMFUniformGrid(cell_name, material_names, xrtm_grid, h5file,
                                          h5filename, h5rtm_grouppath); 

                // Shift the mesh back to local coordinates
                soup.translate(-shift);
              } // for (ixcell)
            } // for (iycell)
          } // for (ixrtm)
        } // for (iyrtm)
      } // for (izlat)
    }  // for (ixasy)
  } // for (iyasy)      

  // Write the XML file
  xdoc.save_file(filepath.c_str(), "  ");

  // Close the HDF5 file
  h5file.close();
} // writeXDMF

//==============================================================================
// write
//==============================================================================

template <std::floating_point T, std::integral I>
void
SpatialPartition<T, I>::write(String const & filename, bool write_kn) const
{
  if (filename.ends_with(".xdmf")) {
    writeXDMF(filename, write_kn);
  } else {
    Log::error("Unsupported file format.");
  }
}

} // namespace um2::mpact
