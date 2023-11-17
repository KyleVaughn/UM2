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

  //  //    int make_cylindrical_pin_mesh(Vector<double> const & radii,
  //  //                                  double const pitch,
  //  //                                  Vector<int> const & num_rings,
  //  //                                  int const num_azimuthal,
  //  //                                  int const mesh_order = 1);
  //  //
  //  //    int make_rectangular_pin_mesh(Vec2<T> const dxdy,
  //  //                                  int const nx,
  //  //                                  int const ny);

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
                LOG_DEBUG("Coarse cell name: " + toString(cell_name));
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
                      static_cast<unsigned char>(cell_materials[iface]));
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
                            static_cast<unsigned char>(cell_materials[iface]));
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
                        T const mcl = quad[mesh_id].getFace(iface).meanChordLength();
                        auto const mat_id = static_cast<Size>(
                            static_cast<unsigned char>(cell_materials[iface]));
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
                            static_cast<unsigned char>(cell_materials[iface]));
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
                            static_cast<unsigned char>(cell_materials[iface]));
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
}

} // namespace um2::mpact
