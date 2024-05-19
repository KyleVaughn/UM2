#pragma once

#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/mesh/rectilinear_partition.hpp>
#include <um2/mesh/regular_partition.hpp>
#include <um2/physics/material.hpp>

namespace um2::mpact
{

//==============================================================================
// MPACT MODEL
//==============================================================================
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

class Model
{

public:
  struct CoarseCell {
    Vec2F xy_extents;
    MeshType mesh_type = MeshType::Invalid;
    Int mesh_id = -1;            // index into the corresponding mesh array
    Vector<MatID> material_ids;  // size = mesh.numFaces()

    PURE [[nodiscard]] constexpr auto
    numFaces() const noexcept -> Int
    {
      return material_ids.size();
    }
  };

  using RTM = RectilinearPartition2<Int>;
  using Lattice = RegularPartition2<Int>;
  using Assembly = RectilinearPartition1<Int>;
  using Core = RectilinearPartition2<Int>;

private:

  // Spatial hierarchy
  Core _core;
  Vector<Assembly> _assemblies;     // Unique assemblies
  Vector<Lattice> _lattices;        // Unique lattices
  Vector<RTM> _rtms;                // Unique RTMs
  Vector<CoarseCell> _coarse_cells; // Unique coarse cells

  // Global materials
  Vector<Material> _materials; // Unique materials

  // Coarse cell meshes
  Vector<TriFVM> _tris;     // Unique triangle meshes
  Vector<QuadFVM> _quads;   // Unique quadrilateral meshes
  Vector<Tri6FVM> _tri6s;   // Unique triangle6 meshes
  Vector<Quad8FVM> _quad8s; // Unique quadrilateral8 meshes

public:
  //============================================================================
  // Constructors
  //============================================================================

  constexpr Model() noexcept = default;

  // NOLINTNEXTLINE(google-explicit-constructor) We want to allow implicit conversion
  Model(String const & filename);

  //============================================================================
  // Member access
  //============================================================================

  PURE [[nodiscard]] constexpr auto
  core() const noexcept -> Core const &;

  PURE [[nodiscard]] constexpr auto
  assemblies() const noexcept -> Vector<Assembly> const &;

  PURE [[nodiscard]] constexpr auto
  lattices() const noexcept -> Vector<Lattice> const &;

  PURE [[nodiscard]] constexpr auto
  rtms() const noexcept -> Vector<RTM> const &;

  PURE [[nodiscard]] constexpr auto
  coarseCells() const noexcept -> Vector<CoarseCell> const &;

  PURE [[nodiscard]] constexpr auto
  materials() noexcept -> Vector<Material> &;

  PURE [[nodiscard]] constexpr auto
  materials() const noexcept -> Vector<Material> const &;

  PURE [[nodiscard]] constexpr auto
  triMeshes() const noexcept -> Vector<TriFVM> const &;

  PURE [[nodiscard]] constexpr auto
  quadMeshes() const noexcept -> Vector<QuadFVM> const &;

  PURE [[nodiscard]] constexpr auto
  tri6Meshes() const noexcept -> Vector<Tri6FVM> const &;

  PURE [[nodiscard]] constexpr auto
  quad8Meshes() const noexcept -> Vector<Quad8FVM> const &;

  //============================================================================
  // Capacity
  //============================================================================

  // Number of unique coarse cells
  PURE [[nodiscard]] constexpr auto
  numCoarseCells() const noexcept -> Int;

  // Number of unique RTMs
  PURE [[nodiscard]] constexpr auto
  numRTMs() const noexcept -> Int;

  // Number of unique lattices
  PURE [[nodiscard]] constexpr auto
  numLattices() const noexcept -> Int;

  // Number of unique assemblies
  PURE [[nodiscard]] constexpr auto
  numAssemblies() const noexcept -> Int;

  // Total number of assemblies in the model (including duplicates)
  PURE [[nodiscard]] constexpr auto
  numAssembliesTotal() const noexcept -> Int;

  // Total number of lattices in the model (including duplicates)
  PURE [[nodiscard]] constexpr auto
  numLatticesTotal() const noexcept -> Int;

  // Total number of RTMs in the model (including duplicates)
  PURE [[nodiscard]] constexpr auto
  numRTMsTotal() const noexcept -> Int;

  // Total number of coarse cells in the model (including duplicates)
  PURE [[nodiscard]] constexpr auto
  numCoarseCellsTotal() const noexcept -> Int;

  // Total number of fine cells in the model
  PURE [[nodiscard]] constexpr auto
  numFineCellsTotal() const noexcept -> Int;

  //============================================================================
  // Getters
  //============================================================================

  PURE [[nodiscard]] constexpr auto
  getCoarseCell(Int cc_id) const noexcept -> CoarseCell const &;

  PURE [[nodiscard]] constexpr auto
  getRTM(Int rtm_id) const noexcept -> RTM const &;

  PURE [[nodiscard]] constexpr auto
  getLattice(Int lat_id) const noexcept -> Lattice const &;

  PURE [[nodiscard]] constexpr auto
  getAssembly(Int asy_id) const noexcept -> Assembly const &;

  PURE [[nodiscard]] constexpr auto
  getCore() const noexcept -> Core const &;

  PURE [[nodiscard]] constexpr auto
  getTriMesh(Int mesh_id) const noexcept -> TriFVM const &;

  PURE [[nodiscard]] constexpr auto
  getQuadMesh(Int mesh_id) const noexcept -> QuadFVM const &;

  PURE [[nodiscard]] constexpr auto
  getTri6Mesh(Int mesh_id) const noexcept -> Tri6FVM const &;

  PURE [[nodiscard]] constexpr auto
  getQuad8Mesh(Int mesh_id) const noexcept -> Quad8FVM const &;

  //============================================================================
  // Modifiers
  //============================================================================

  HOSTDEV void
  clear() noexcept;

  auto
  addMaterial(Material const & material, bool validate = true) -> Int;

  auto
  addCylindricalPinMesh(Float pitch,
                        Vector<Float> const & radii,
                        Vector<Int> const & num_rings,
                        Int num_azimuthal,
                        Int mesh_order = 1) -> Int;

  auto
  addRectangularPinMesh(Vec2F xy_extents, Int nx_faces, Int ny_faces) -> Int;

  auto
  addTriMesh(TriFVM const & mesh) -> Int;

  auto
  addQuadMesh(QuadFVM const & mesh) -> Int;

  auto
  addTri6Mesh(Tri6FVM const & mesh) -> Int;

  auto
  addQuad8Mesh(Quad8FVM const & mesh) -> Int;

  auto
  addCylindricalPinCell(Float pitch,
                        Vector<Float> const & radii,
                        Vector<Material> const & materials,
                        Vector<Int> const & num_rings,
                        Int num_azimuthal,
                        Int mesh_order = 1) -> Int;

  auto
  addCoarseCell(Vec2F xy_extents,
      MeshType mesh_type = MeshType::Invalid,
      Int mesh_id = -1,
      Vector<MatID> const & material_ids = {}) -> Int;

  auto
  addRTM(Vector<Vector<Int>> const & cc_ids) -> Int;

  auto
  addLattice(Vector<Vector<Int>> const & rtm_ids) -> Int;

#if UM2_ENABLE_FLOAT64
  auto
  addAssembly(Vector<Int> const & lat_ids, Vector<Float> const & z = {-0.5, 0.5}) -> Int;
#else
  auto
  addAssembly(Vector<Int> const & lat_ids, Vector<Float> const & z = {-0.5F, 0.5F}) -> Int;
#endif

  auto
  addCore(Vector<Vector<Int>> const & asy_ids) -> Int;

  // A convenience function to cover an area with a regular grid of coarse cells.
  // This method maps each coarse cell to a unique RTM and places all the RTMs in a lattice.
  void
  addCoarseGrid(Vec2F xy_extents, Vec2I xy_num_cells);

  void
  importCoarseCellMeshes(String const & filename);

  //============================================================================
  // Methods
  //============================================================================

  // This can be very slow for large models. Used for debugging.
  explicit operator PolytopeSoup() const noexcept;

  void
  read(String const & filename);

  void
  write(String const & filename,
      bool write_knudsen_data = false, 
      bool write_xsec_data = false) const;

  // Return a vector of group-wise optical thicknesses for the coarse cell.
  //void 
  //getCoarseCellOpticalThickness(Int cc_id, Vector<Float> & taus) const;

  // Homogenize the material and return the resulting cross section.
  // for each face i in 0, 1, ... num_faces - 1, 
  // Sigma_x = (sum_{i} A_i * Sigma_x_i) / sum_{i} A_i
  PURE auto
  getCoarseCellHomogenizedXSec(Int cc_id) const -> XSec;

  void
  writeCMFDInfo(String const & filename) const;

}; // struct Model

//=============================================================================
// Free functions
//=============================================================================

// We need to get labels like "Coarse_Cell_00001" or "Assembly_00021". Instead
// of importing stringstream or doing string concatenation, we can use this
// function.
template <typename Str>
inline void
incrementASCIINumber(Str & str)
{
  // '0' to '9' are contiguous in ASCII
  // '0' = 48, '9' = 57
  // While the back character is '9', set it to '0',
  // move p to the next character to the left, and increment it.
  ASSERT(!str.empty());
  char * p = str.data() + str.size() - 1;
  while (*p == '9') {
    *p-- = '0';
  }
  ASSERT(p >= str.data());
  ASSERT('0' <= *p);
  ASSERT(*p <= '9');
  ++(*p);
}

inline auto
getASCIINumber(Int num)
{
  ASSERT(num >= 0);
  ASSERT(num < 100000);
  String str("00000"); // technically only need
  char * p = str.end() - 1;
  for (Int i = 0; i < 5; ++i) {
    *p-- = static_cast<int8_t>(num % 10 + 48);
    num /= 10;
  }
  return str;
}

//=============================================================================
// Member access
//=============================================================================

PURE [[nodiscard]] constexpr auto
Model::core() const noexcept -> Core const &
{
  return _core;
}

PURE [[nodiscard]] constexpr auto
Model::assemblies() const noexcept -> Vector<Assembly> const &
{
  return _assemblies;
}

PURE [[nodiscard]] constexpr auto
Model::lattices() const noexcept -> Vector<Lattice> const &
{
  return _lattices;
}

PURE [[nodiscard]] constexpr auto
Model::rtms() const noexcept -> Vector<RTM> const &
{
  return _rtms;
}

PURE [[nodiscard]] constexpr auto
Model::coarseCells() const noexcept -> Vector<CoarseCell> const &
{
  return _coarse_cells;
}

PURE [[nodiscard]] constexpr auto
Model::materials() noexcept -> Vector<Material> &
{
  return _materials;
}

PURE [[nodiscard]] constexpr auto
Model::materials() const noexcept -> Vector<Material> const &
{
  return _materials;
}

PURE [[nodiscard]] constexpr auto
Model::triMeshes() const noexcept -> Vector<TriFVM> const &
{
  return _tris;
}

PURE [[nodiscard]] constexpr auto
Model::quadMeshes() const noexcept -> Vector<QuadFVM> const &
{
  return _quads;
}

PURE [[nodiscard]] constexpr auto
Model::tri6Meshes() const noexcept -> Vector<Tri6FVM> const &
{
  return _tri6s;
}

PURE [[nodiscard]] constexpr auto
Model::quad8Meshes() const noexcept -> Vector<Quad8FVM> const &
{
  return _quad8s;
}

//=============================================================================
// Capacity
//=============================================================================

PURE [[nodiscard]] constexpr auto
Model::numCoarseCells() const noexcept -> Int
{
  return _coarse_cells.size();
}

PURE [[nodiscard]] constexpr auto
Model::numRTMs() const noexcept -> Int
{
  return _rtms.size();
}

PURE [[nodiscard]] constexpr auto
Model::numLattices() const noexcept -> Int
{
  return _lattices.size();
}

PURE [[nodiscard]] constexpr auto
Model::numAssemblies() const noexcept -> Int
{
  return _assemblies.size();
}

PURE [[nodiscard]] constexpr auto
Model::numAssembliesTotal() const noexcept -> Int
{
  return _core.children().size();
}

PURE [[nodiscard]] constexpr auto
Model::numLatticesTotal() const noexcept -> Int
{
  Int total = 0;
  for (auto const & asy_id : _core.children()) {
    total += _assemblies[asy_id].children().size();
  }
  return total;
}

PURE [[nodiscard]] constexpr auto
Model::numRTMsTotal() const noexcept -> Int
{
  Int total = 0;
  for (auto const & asy_id : _core.children()) {
    for (auto const & lat_id : _assemblies[asy_id].children()) {
      total += _lattices[lat_id].children().size();
    }
  }
  return total;
}

PURE [[nodiscard]] constexpr auto
Model::numCoarseCellsTotal() const noexcept -> Int
{
  Int total = 0;
  for (auto const & asy_id : _core.children()) {
    for (auto const & lat_id : _assemblies[asy_id].children()) {
      for (auto const & rtm_id : _lattices[lat_id].children()) {
        total += _rtms[rtm_id].children().size();
      }
    }
  }
  return total;
}

PURE [[nodiscard]] constexpr auto
Model::numFineCellsTotal() const noexcept -> Int
{
  Int total = 0;
  for (auto const & asy_id : _core.children()) {
    for (auto const & lat_id : _assemblies[asy_id].children()) {
      for (auto const & rtm_id : _lattices[lat_id].children()) {
        for (auto const & cc_id : _rtms[rtm_id].children()) {
          total += _coarse_cells[cc_id].numFaces();
        }
      }
    }
  }
  return total;
}

//=============================================================================
// Getters
//=============================================================================

PURE [[nodiscard]] constexpr auto
Model::getCoarseCell(Int cc_id) const noexcept -> CoarseCell const &
{
  return _coarse_cells[cc_id];
}

PURE [[nodiscard]] constexpr auto
Model::getRTM(Int rtm_id) const noexcept -> RTM const &
{
  return _rtms[rtm_id];
}

PURE [[nodiscard]] constexpr auto
Model::getLattice(Int lat_id) const noexcept -> Lattice const &
{
  return _lattices[lat_id];
}

PURE [[nodiscard]] constexpr auto
Model::getAssembly(Int asy_id) const noexcept -> Assembly const &
{
  return _assemblies[asy_id];
}

PURE [[nodiscard]] constexpr auto
Model::getCore() const noexcept -> Core const &
{
  return _core;
}

PURE [[nodiscard]] constexpr auto
Model::getTriMesh(Int const mesh_id) const noexcept -> TriFVM const &
{
  return _tris[mesh_id];
}

PURE [[nodiscard]] constexpr auto
Model::getQuadMesh(Int const mesh_id) const noexcept -> QuadFVM const &
{
  return _quads[mesh_id];
}

PURE [[nodiscard]] constexpr auto
Model::getTri6Mesh(Int const mesh_id) const noexcept -> Tri6FVM const &
{
  return _tri6s[mesh_id];
}

PURE [[nodiscard]] constexpr auto
Model::getQuad8Mesh(Int const mesh_id) const noexcept -> Quad8FVM const &
{
  return _quad8s[mesh_id];
}

} // namespace um2::mpact
