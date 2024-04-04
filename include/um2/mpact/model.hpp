#pragma once

#include <um2/mesh/face_vertex_mesh.hpp>
//#include <um2/mesh/rectilinear_partition.hpp>
//#include <um2/mesh/regular_partition.hpp>
//#include <um2/physics/material.hpp>

namespace um2::mpact
{

//==============================================================================
// MPACT MODEL 
//==============================================================================
///// An equivalent representation to the various mesh hierarchies in an MPACT model.
/////
/////  ************************
/////  *****VERY IMPORTANT*****
/////  ************************
/////  - The pin mesh coordinate system origin in MPACT is the center of the pin. Here
/////    we use the bottom left corner of the pin mesh as the origin.
/////  - In MPACT, two pins with the same mesh but different heights are considered
/////    different meshes. Here we consider them the same mesh.
/////
///// The MPACT spatial partition consists of:
/////      1. Core
/////          A rectilinear partition of the XY-domain into assemblies. The assemblies
/////          must have the same start and stop heights.
/////      2. Assembly
/////          A rectilinear partition of the Z-domain into 2D axial slices (lattices).
/////      3. Lattice
/////          A regular partition of the XY-domain into equal-sized axis-aligned
/////          rectangles, also known as "ray tracing modules" (RTMs).
/////          Each lattice has a local coordinate system with (0, 0) in the bottom
/////          left corner.
/////      4. RTM
/////          A rectilinear partition of the XY-domain into coarse cells.
/////          Every RTM is exactly the same width and height in all lattices.
/////          This property is a necessity for modular ray tracing.
/////          Each RTM has a local coordinate system with (0, 0) in the bottom
/////          left corner.
/////      5. Coarse cell
/////          A 2D axis-aligned box (AABB), containing a mesh which completely
/////          fills the box's interior. This mesh is the "fine mesh". It is made
/////          up of fine cells (triangles, quadrilaterals, etc.). Each of these
/////          fine cells has an integer material ID. This structure is
/////          represented as a fine mesh ID and a material ID list ID, allowing the
/////          same mesh to be reused for multiple pins with different materials.
/////          Each coarse cell has a local coordinate system with (0, 0) in the
/////          bottom left corner.
/////
/////          In MPACT, the coarse cells typically contain the geometry for a single
/////          pin, centered in middle of the coarse cell - hence the name "pin cell".
/////          In this code, due to the arbitrary nature of the geometry, the coarse
/////          cells may contain a piece of a pin, multiple pins, or any other
/////          arbitrary geometry.
/////

class Model 
{

public:
//  struct CoarseCell {
//    Vec2<F> dxdy; // dx, dy
//    MeshType mesh_type = MeshType::None;
//    Int mesh_id = -1;               // index into the corresponding mesh array
//    Vector<MaterialID> material_ids; // size = mesh.numFaces()
//
//    PURE [[nodiscard]] constexpr auto
//    numFaces() const noexcept -> Int
//    {
//      return material_ids.size();
//    }
//  };
//
//  using RTM = RectilinearPartition2<I>;
//  using Lattice = RegularPartition2<I>;
//  using Assembly = RectilinearPartition1<I>;
//  using Core = RectilinearPartition2<I>;
//
private:
//  // The children IDs are used to index the corresponding array.
//  // Child ID = -1 indicates that the child does not exist. This is used
//  // for when the child should be generated automatically.
//
//  Core _core;
//  Vector<Assembly> _assemblies;
//  Vector<Lattice> _lattices;
//  Vector<RTM> _rtms;
//  Vector<CoarseCell> _coarse_cells;
//
//  Vector<Material> _materials;
//
//  Vector<TriFVM> _tris;
  Vector<QuadFVM> _quads;
//  Vector<Tri6FVM> _tri6s;
  Vector<Quad8FVM> _quad8s;

public:
  //============================================================================
  // Constructors
  //============================================================================

  constexpr Model() noexcept = default;

//  //============================================================================
//  // Accessors
//  //============================================================================
//
//  PURE [[nodiscard]] constexpr auto
//  numCoarseMeshes() const noexcept -> Int;
//
//  PURE [[nodiscard]] constexpr auto
//  numCoarseCells() const noexcept -> Int;
//
//  PURE [[nodiscard]] constexpr auto
//  numRTMs() const noexcept -> Int;
//
//  PURE [[nodiscard]] constexpr auto
//  numLattices() const noexcept -> Int;
//
//  PURE [[nodiscard]] constexpr auto
//  numAssemblies() const noexcept -> Int;
//
//  PURE [[nodiscard]] constexpr auto
//  getCoarseCell(Int cc_id) const noexcept -> CoarseCell const &;
//
//  PURE [[nodiscard]] constexpr auto
//  getRTM(Int rtm_id) const noexcept -> RTM const &;
//
//  PURE [[nodiscard]] constexpr auto
//  getLattice(Int lat_id) const noexcept -> Lattice const &;
//
//  PURE [[nodiscard]] constexpr auto
//  getAssembly(Int asy_id) const noexcept -> Assembly const &;
//
//  PURE [[nodiscard]] constexpr auto
//  getCore() const noexcept -> Core const &;
//
//  PURE [[nodiscard]] auto
//  getTriMesh(Int mesh_id) const noexcept -> TriFVM const &;
//
  PURE [[nodiscard]] constexpr auto
  getQuadMesh(Int mesh_id) const noexcept -> QuadFVM const &;

//  PURE [[nodiscard]] auto
//  getTri6Mesh(Int mesh_id) const noexcept -> Tri6FVM const &;

  PURE [[nodiscard]] constexpr auto
  getQuad8Mesh(Int mesh_id) const noexcept -> Quad8FVM const &;

//  //============================================================================
//  // Methods
//  //============================================================================
//
//  HOSTDEV void
//  clear() noexcept;
//
//  void
//  checkMeshExists(MeshType mesh_type, Int mesh_id) const;
//
//  auto
//  addMaterial(Material const & material) -> Int;
//
  auto
  addCylindricalPinMesh(Float pitch, 
                        Vector<Float> const & radii, 
                        Vector<Int> const & num_rings,
                        Int num_azimuthal, 
                        Int mesh_order = 1) -> Int;

  auto
  addRectangularPinMesh(Vec2F xy_extents, Int nx_faces, Int ny_faces) -> Int;
//
//  auto
//  addCoarseCell(Vec2<F> dxdy, MeshType mesh_type = MeshType::None, Int mesh_id = -1,
//                 Vector<MaterialID> const & material_ids = {}) -> Int;
//
//  auto
//  addRTM(Vector<Vector<I>> const & cc_ids) -> Int;
//
//  auto
//  addLattice(Vector<Vector<I>> const & rtm_ids) -> Int;
//  //
//  //  //  auto
//  //  //  stdMakeLattice(std::vector<std::vector<I>> const & rtm_ids) -> Int;
//  //
//  auto
//  addAssembly(Vector<I> const & lat_ids, Vector<F> const & z = {-1, 1}) -> Int;
//
//  auto
//  addCore(Vector<Vector<I>> const & asy_ids) -> Int;
//  //
//  //  //  auto
//  //  //  stdMakeCore(std::vector<std::vector<I>> const & asy_ids) -> Int;
//
//  // Import coarse cells and pin meshes from a file.
//  void
//  importCoarseCells(String const & filename);
//
//  // Assume everything fits in one of the next highest hierarchy levels.
//  void
//  fillHierarchy();
//
//  //  //  void
//  //  //  toPolytopeSoup(PolytopeSoup & soup, bool write_kn = false) const;
//  //  //
//  //  //  void
//  //  //  getMaterialNames(Vector<String> & material_names) const;
//  //  //
//  //  //  void
//  //  //  write(String const & filename, bool write_kn = false) const;
//  //  //
//  //  //  void
//  //  //  writeXDMF(String const & filepath, bool write_kn = false) const;
//  //  //
}; // struct Model
//
////=============================================================================
//// Accessors
////=============================================================================
//
//PURE [[nodiscard]] constexpr auto
//Model::numCoarseMeshes() const noexcept -> Int
//{
//  return _tris.size() + _quads.size() + _tri6s.size() + _quad8s.size();
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::numCoarseCells() const noexcept -> Int
//{
//  return _coarse_cells.size();
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::numRTMs() const noexcept -> Int
//{
//  return _rtms.size();
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::numLattices() const noexcept -> Int
//{
//  return _lattices.size();
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::numAssemblies() const noexcept -> Int
//{
//  return _assemblies.size();
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::getCoarseCell(Int cc_id) const noexcept -> CoarseCell const &
//{
//  return _coarse_cells[cc_id];
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::getRTM(Int rtm_id) const noexcept -> RTM const &
//{
//  return _rtms[rtm_id];
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::getLattice(Int lat_id) const noexcept -> Lattice const &
//{
//  return _lattices[lat_id];
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::getAssembly(Int asy_id) const noexcept -> Assembly const &
//{
//  return _assemblies[asy_id];
//}
//
//PURE [[nodiscard]] constexpr auto
//Model::getCore() const noexcept -> Core const &
//{
//  return _core;
//}


PURE [[nodiscard]] constexpr auto
Model::getQuadMesh(Int const mesh_id) const noexcept -> QuadFVM const &
{
  return _quads[mesh_id];
}

PURE [[nodiscard]] constexpr auto
Model::getQuad8Mesh(Int const mesh_id) const noexcept -> Quad8FVM const &
{
  return _quad8s[mesh_id];
}

} // namespace um2::mpact
