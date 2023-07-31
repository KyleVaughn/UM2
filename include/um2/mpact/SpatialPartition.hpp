#pragma once

#include <um2/common/Vector.hpp>
// #include <um2/physics/material.hpp>
#include <um2/mesh/RectilinearPartition.hpp>
#include <um2/mesh/RegularPartition.hpp>
// #include <um2/mesh/face_vertex_mesh.hpp>
// #include <um2/mesh/io.hpp>
// #include <um2/ray_casting/intersect/ray-linear_polygon_mesh.hpp>
// #include <um2/ray_casting/intersect/ray-quadratic_polygon_mesh.hpp>

// #include <iomanip>

// #include <thrust/pair.h> // thrust::pair

// #include <string> // std::string

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

  //    // Take this out of the struct?
  //    struct CoarseCell {
  //        Vec2<T> dxdy; // dx, dy
  //        I mesh_type = -1;  // see MeshType
  //        I mesh_id = -1;    // index into the corresponding mesh array
  //        Vector<MaterialID> material_ids; // size = num_faces(mesh)
  //
  //        UM2_PURE constexpr length_t num_faces() const { return material_ids.size(); }
  //
  //    };
  typedef RectilinearPartition2<T, I> RTM;
  typedef RegularPartition2<T, I> Lattice;
  typedef RectilinearPartition1<T, I> Assembly;

  // The children IDs are used to index the corresponding array.
  // Child ID = -1 indicates that the child does not exist. This is used
  // for when the child should be generated automatically.

  RectilinearPartition2<T, I> core;
  Vector<Assembly> assemblies;
  Vector<Lattice> lattices;
  Vector<RTM> rtms;
  Vector<CoarseCell> coarse_cells;

  Vector<Material> materials;

  //    Vector<TriMesh<T, I>> tri;
  //    Vector<QuadMesh<T, I>> quad;
  //    Vector<TriQuadMesh<T, I>> tri_quad;
  //    Vector<QuadraticTriMesh<T, I>> quadratic_tri;
  //    Vector<QuadraticQuadMesh<T, I>> quadratic_quad;
  //    Vector<QuadraticTriQuadMesh<T, I>> quadratic_tri_quad;

  // -- Constructors --

  //    UM2_HOSTDEV SpatialPartition() = default;
  //
  //    UM2_HOSTDEV void clear();

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
  //    int make_coarse_cell(Vec2<T> const dxdy,
  //                         I const mesh_type = -1,
  //                         I const mesh_id = -1,
  //                         MaterialID const * const material_ids = nullptr,
  //                         length_t const num_faces = 0);
  //
  //    int make_rtm(std::vector<std::vector<int>> const & cc_ids);
  //
  //    int make_lattice(std::vector<std::vector<int>> const & rtm_ids);
  //
  //    int make_assembly(std::vector<int> const & lat_ids,
  //                      std::vector<double> const & z = {-1, 1});
  //
  //    int make_core(std::vector<std::vector<int>> const & ass_ids);
  //
  //    int import_coarse_cells(std::string const & filename);
  //
  //    void coarse_cell_heights(Vector<std::pair<int, double>> & id_dz) const;
  //
  //    void coarse_cell_face_areas(length_t const cc_id, Vector<T> & areas) const;
  //
  //    length_t coarse_cell_find_face(length_t const cc_id, Point2<T> const & p) const;
  //
  //    Point2<T> coarse_cell_face_centroid(length_t const cc_id, length_t const face_id)
  //    const;
  //
  //    void intersect_coarse_cell(length_t const cc_id,
  //                               Ray2<T> const & ray,
  //                               Vector<T> & intersections) const;
  //
  //
  //    void intersect_coarse_cell(length_t const cc_id, // Fixed-size buffer
  //                               Ray2<T> const & ray,
  //                               T * const intersections,
  //                               int * const n) const;
  //
  //
  //    void rtm_heights(Vector<std::pair<int, double>> & id_dz) const;
  //
  //    void lattice_heights(Vector<std::pair<int, double>> & id_dz) const;
  //
  //    void coarse_cell_face_data(length_t const cc_id,
  //                               length_t * const mesh_type,
  //                               length_t * const num_vertices,
  //                               length_t * const num_faces,
  //                               T ** const vertices,
  //                               I ** const fv_offsets,
  //                               I ** const fv) const;

}; // struct SpatialPartition

// template <std::floating_point T, std::signed_integral I>
// UM2_PURE constexpr int num_unique_coarse_cells(SpatialPartition<T, I> const & sp)
//{
//     return static_cast<int>(sp.coarse_cells.size());
// }
//
// template <std::floating_point T, std::signed_integral I>
// UM2_PURE constexpr int num_unique_rtms(SpatialPartition<T, I> const & sp)
//{
//     return static_cast<int>(sp.rtms.size());
// }
//
// template <std::floating_point T, std::signed_integral I>
// UM2_PURE constexpr int num_unique_lattices(SpatialPartition<T, I> const & sp)
//{
//     return static_cast<int>(sp.lattices.size());
// }
//
// template <std::floating_point T, std::signed_integral I>
// UM2_PURE constexpr int num_unique_assemblies(SpatialPartition<T, I> const & sp)
//{
//     return static_cast<int>(sp.assemblies.size());
// }

} // namespace um2::mpact

#include "SpatialPartition.inl"
