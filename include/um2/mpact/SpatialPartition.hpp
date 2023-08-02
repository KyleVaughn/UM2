#pragma once

#include <um2/common/Log.hpp>
#include <um2/common/Vector.hpp>
#include <um2/mesh/MeshType.hpp>
#include <um2/mesh/QuadMesh.hpp>
#include <um2/mesh/QuadraticQuadMesh.hpp>
#include <um2/mesh/QuadraticTriMesh.hpp>
#include <um2/mesh/RectilinearPartition.hpp>
#include <um2/mesh/RegularPartition.hpp>
#include <um2/mesh/TriMesh.hpp>
#include <um2/physics/Material.hpp>
// #include <um2/mesh/io.hpp>
// #include <um2/ray_casting/intersect/ray-linear_polygon_mesh.hpp>
// #include <um2/ray_casting/intersect/ray-quadratic_polygon_mesh.hpp>

// #include <iomanip>

// #include <thrust/pair.h> // thrust::pair

#include <string>

namespace um2::mpact
{

// -----------------------------------------------------------------------------
// MPACT SPATIAL PARTITON
// -----------------------------------------------------------------------------
// An equivalent representation to the various mesh hierarchies in an MPACT model.
//
//  ************************
//  *****VERY IMPORTANT*****
//  ************************
//  - The pin mesh coordinate system origin in MPACT is the center of the pin. Here
//    we use the bottom left corner of the pin mesh as the origin.
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
template <std::floating_point T, std::signed_integral I>
struct SpatialPartition {

  // Take this out of the struct?
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

  Vector<Material> materials;

  Vector<TriMesh<2, T, I>> tri;
  Vector<QuadMesh<2, T, I>> quad;
  // Vector<TriQuadMesh<T, I>> tri_quad;
  Vector<QuadraticTriMesh<2, T, I>> quadratic_tri;
  Vector<QuadraticQuadMesh<2, T, I>> quadratic_quad;
  // Vector<QuadraticTriQuadMesh<T, I>> quadratic_tri_quad;

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

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr void
  clear() noexcept;

  //    int make_cylindrical_pin_mesh(std::vector<double> const & radii,
  //                                  double const pitch,
  //                                  std::vector<int> const & num_rings,
  //                                  int const num_azimuthal,
  //                                  int const mesh_order = 1);
  //
  //    int make_rectangular_pin_mesh(Vec2<T> const dxdy,
  //                                  int const nx,
  //                                  int const ny);
  //
  //    int make_coarse_cell(I const mesh_type,
  //                         I const mesh_id,
  //                         std::vector<Material> const & materials);
  //
  auto
  makeCoarseCell(Vec2<T> dxdy, MeshType mesh_type = MeshType::None, Size mesh_id = -1,
                 Vector<MaterialID> const & material_ids = {}) -> Size;

  auto
  makeRTM(std::vector<std::vector<Size>> const & cc_ids) -> Size;

  auto
  makeLattice(std::vector<std::vector<Size>> const & rtm_ids) -> Size;

  auto 
  makeAssembly(std::vector<Size> const & lat_ids,
               std::vector<T> const & z = {-1, 1});
  //
  //    int make_core(std::vector<std::vector<int>> const & ass_ids);
  //
  //    int import_coarse_cells(std::string const & filename);
  //
  //    void coarse_cell_heights(Vector<std::pair<int, double>> & id_dz) const;
  //
  //    void coarse_cell_face_areas(Size const cc_id, Vector<T> & areas) const;
  //
  //    Size coarse_cell_find_face(Size const cc_id, Point2<T> const & p) const;
  //
  //    Point2<T> coarse_cell_face_centroid(Size const cc_id, Size const face_id)
  //    const;
  //
  //    void intersect_coarse_cell(Size const cc_id,
  //                               Ray2<T> const & ray,
  //                               Vector<T> & intersections) const;
  //
  //
  //    void intersect_coarse_cell(Size const cc_id, // Fixed-size buffer
  //                               Ray2<T> const & ray,
  //                               T * const intersections,
  //                               int * const n) const;
  //
  //
  //    void rtm_heights(Vector<std::pair<int, double>> & id_dz) const;
  //
  //    void lattice_heights(Vector<std::pair<int, double>> & id_dz) const;
  //
  //    void coarse_cell_face_data(Size const cc_id,
  //                               Size * const mesh_type,
  //                               Size * const num_vertices,
  //                               Size * const num_faces,
  //                               T ** const vertices,
  //                               I ** const fv_offsets,
  //                               I ** const fv) const;

}; // struct SpatialPartition

} // namespace um2::mpact

#include "SpatialPartition.inl"
