#pragma once

#include <um2/mesh/FaceVertexMesh.hpp>
#include <um2/mesh/RectilinearPartition.hpp>
#include <um2/mesh/RegularPartition.hpp>
#include <um2/mesh/io.hpp>
#include <um2/physics/Material.hpp>

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
struct SpatialPartition {

  // Take this out of the struct?
  struct CoarseCell {
    Vec2<Float> dxdy; // dx, dy
    MeshType mesh_type = MeshType::None;
    Size mesh_id = -1;               // index into the corresponding mesh array
    Vector<MaterialID> material_ids; // size = mesh.numFaces()

    PURE [[nodiscard]] constexpr auto
    numFaces() const noexcept -> Size
    {
      return material_ids.size();
    }
  };
  using RTM = RectilinearPartition2<Float, Int>;
  using Lattice = RegularPartition2<Float, Int>;
  using Assembly = RectilinearPartition1<Float, Int>;

  // The children IDs are used to index the corresponding array.
  // Child ID = -1 indicates that the child does not exist. This is used
  // for when the child should be generated automatically.

  RectilinearPartition2<Float, Int> core;
  Vector<Assembly> assemblies;
  Vector<Lattice> lattices;
  Vector<RTM> rtms;
  Vector<CoarseCell> coarse_cells;

  Vector<Material> materials;

  Vector<TriMesh<2, Float, Int>> tri;
  Vector<QuadMesh<2, Float, Int>> quad;
  Vector<QuadraticTriMesh<2, Float, Int>> quadratic_tri;
  Vector<QuadraticQuadMesh<2, Float, Int>> quadratic_quad;

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr SpatialPartition() noexcept = default;

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE [[nodiscard]] constexpr auto
  numCoarseCells() const noexcept -> Size
  {
    return coarse_cells.size();
  }

  PURE [[nodiscard]] constexpr auto
  numRTMs() const noexcept -> Size
  {
    return rtms.size();
  }

  PURE [[nodiscard]] constexpr auto
  numLattices() const noexcept -> Size
  {
    return lattices.size();
  }

  PURE [[nodiscard]] constexpr auto
  numAssemblies() const noexcept -> Size
  {
    return assemblies.size();
  }

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr void
  clear() noexcept
  {
    core.clear();
    assemblies.clear();
    lattices.clear();
    rtms.clear();
    coarse_cells.clear();

    tri.clear();
    quad.clear();
    quadratic_tri.clear();
    quadratic_quad.clear();
  }

  inline void
  checkMeshExists(MeshType mesh_type, Size mesh_id) const
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

  //    int make_cylindrical_pin_mesh(Vector<double> const & radii,
  //                                  double const pitch,
  //                                  Vector<int> const & num_rings,
  //                                  int const num_azimuthal,
  //                                  int const mesh_order = 1);
  //
  //    int make_rectangular_pin_mesh(Vec2<Float> const dxdy,
  //                                  int const nx,
  //                                  int const ny);
  //
  auto
  makeCoarseCell(Vec2<Float> dxdy, MeshType mesh_type = MeshType::None, Size mesh_id = -1,
                 Vector<MaterialID> const & material_ids = {}) -> Size;

  auto
  makeRTM(Vector<Vector<Size>> const & cc_ids) -> Size;

  auto
  makeLattice(Vector<Vector<Size>> const & rtm_ids) -> Size;

  auto
  makeAssembly(Vector<Size> const & lat_ids, Vector<Float> const & z = {-1, 1})
      -> Size;

  auto
  makeCore(Vector<Vector<Size>> const & asy_ids) -> Size;

  void
  importCoarseCells(String const & filename);

}; // struct SpatialPartition

} // namespace um2::mpact
