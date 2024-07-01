#include <um2.hpp>
#include <um2/common/cast_if_not.hpp>
#include <um2/common/logger.hpp>
#include <um2/config.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/math/vec.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/mpact/model.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/math/abs.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/utility/pair.hpp>
#include <um2/stdlib/vector.hpp>
#include <um2c.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>

//==============================================================================
// Data sizes
//==============================================================================

void
um2SizeOfInt(int * size)
{
  *size = static_cast<int>(sizeof(Int));
}

void
um2SizeOfFloat(int * size)
{
  *size = static_cast<int>(sizeof(Float));
}

//==============================================================================
// um2Malloc and um2Free
//==============================================================================

void
um2Malloc(void ** const p, Int const size)
{
  *p = malloc(static_cast<size_t>(size));
}

void
um2Free(void * const p)
{
  free(p);
}

//==============================================================================
// um2Initialize and um2Finalize
//==============================================================================

void
um2Initialize()
{
  um2::initialize();
}

void
um2Finalize()
{
  um2::finalize();
}

//==============================================================================
// MPACT Model
//==============================================================================

void
um2NewMPACTModel(void ** const model)
{
  *model = reinterpret_cast<void *>(new um2::mpact::Model());
}

void
um2DeleteMPACTModel(void * const model)
{
  delete reinterpret_cast<um2::mpact::Model *>(model);
}

void
um2ReadMPACTModel(char const * const path, void ** const model)
{
  um2::String const path_str(path);
  *model = reinterpret_cast<void *>(new um2::mpact::Model());
  auto & sp = *reinterpret_cast<um2::mpact::Model *>(*model);
  sp.read(path_str);
}

// Num
//------------------------------------------------------------------------------

void
um2MPACTNumCoarseCells(void * const model, Int * const n)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *n = sp.numCoarseCells();
  ASSERT(*n > 0);
}

void
um2MPACTNumRTMs(void * const model, Int * const n)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *n = sp.numRTMs();
  ASSERT(*n > 0);
}

void
um2MPACTNumLattices(void * const model, Int * const n)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *n = sp.numLattices();
  ASSERT(*n > 0);
}

void
um2MPACTNumAssemblies(void * const model, Int * const n)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *n = sp.numAssemblies();
  ASSERT(*n > 0);
}

// NumCells
//------------------------------------------------------------------------------

void
um2MPACTCoreNumCells(void * const model, Int * const nx, Int * const ny)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const ncells = sp.core().grid().numCells();
  *nx = ncells[0];
  *ny = ncells[1];
  ASSERT(*nx > 0);
  ASSERT(*ny > 0);
}

void
um2MPACTAssemblyNumCells(void * const model, Int const asy_id, Int * const nx)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *nx = sp.getAssembly(asy_id).grid().numCells(0);
  ASSERT(*nx > 0);
}

void
um2MPACTLatticeNumCells(void * const model, Int const lat_id, Int * const nx,
                        Int * const ny)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const ncells = sp.getLattice(lat_id).grid().numCells();
  *nx = ncells[0];
  *ny = ncells[1];
  ASSERT(*nx > 0);
  ASSERT(*ny > 0);
}

void
um2MPACTRTMNumCells(void * const model, Int const rtm_id, Int * const nx, Int * const ny)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const ncells = sp.getRTM(rtm_id).grid().numCells();
  *nx = ncells[0];
  *ny = ncells[1];
  ASSERT(*nx > 0);
  ASSERT(*ny > 0);
}

// GetChild
//------------------------------------------------------------------------------

void
um2MPACTCoreGetChild(void * const model, Int const ix, Int const iy, Int * const child)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *child = sp.core().getChild(ix, iy);
  ASSERT(*child >= 0);
}

void
um2MPACTAssemblyGetChild(void * const model, Int const asy_id, Int const ix,
                         Int * const child)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *child = sp.getAssembly(asy_id).getChild(ix);
  ASSERT(*child >= 0);
}

void
um2MPACTLatticeGetChild(void * const model, Int const lat_id, Int const ix, Int const iy,
                        Int * const child)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *child = sp.getLattice(lat_id).getChild(ix, iy);
  ASSERT(*child >= 0);
}

void
um2MPACTRTMGetChild(void * const model, Int const rtm_id, Int const ix, Int const iy,
                    Int * const child)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *child = sp.getRTM(rtm_id).getChild(ix, iy);
  ASSERT(*child >= 0);
}

//==============================================================================
// CoarseCell
//==============================================================================

void
um2MPACTCoarseCellNumFaces(void * const model, Int const cc_id, Int * const num_faces)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *num_faces = sp.getCoarseCell(cc_id).numFaces();
  ASSERT(*num_faces > 0);
}

void
um2MPACTCoarseCellWidth(void * const model, Int const cc_id, Float * const width)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *width = sp.getCoarseCell(cc_id).xy_extents[0];
  ASSERT(*width > 0);
}

void
um2MPACTCoarseCellHeight(void * const model, Int const cc_id, Float * const height)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *height = sp.getCoarseCell(cc_id).xy_extents[1];
}

void
um2MPACTCoarseCellFaceAreas(void * const model, Int const cc_id, Float * const areas)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const & cc = sp.getCoarseCell(cc_id);
  switch (cc.mesh_type) {
  case um2::MeshType::Tri: {
    auto const & mesh = sp.getTriMesh(cc.mesh_id);
    for (Int i = 0; i < mesh.numFaces(); ++i) {
      areas[i] = mesh.getFace(i).area();
    }
    break;
  }
  case um2::MeshType::Quad: {
    auto const & mesh = sp.getQuadMesh(cc.mesh_id);
    for (Int i = 0; i < mesh.numFaces(); ++i) {
      areas[i] = mesh.getFace(i).area();
    }
    break;
  }
  case um2::MeshType::QuadraticTri: {
    auto const & mesh = sp.getTri6Mesh(cc.mesh_id);
    for (Int i = 0; i < mesh.numFaces(); ++i) {
      areas[i] = mesh.getFace(i).area();
    }
    break;
  }
  case um2::MeshType::QuadraticQuad: {
    auto const & mesh = sp.getQuad8Mesh(cc.mesh_id);
    for (Int i = 0; i < mesh.numFaces(); ++i) {
      areas[i] = mesh.getFace(i).area();
    }
    break;
  }
  default:
    LOG_ERROR("Invalid mesh type");
    return;
  }
}

void
um2MPACTCoarseCellFaceContaining(void * const model, Int const cc_id, Float const x,
                                 Float const y, Int * const face_id)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const & cc = sp.getCoarseCell(cc_id);
  switch (cc.mesh_type) {
  case um2::MeshType::Tri: {
    *face_id = sp.getTriMesh(cc.mesh_id).faceContaining(um2::Point2F(x, y));
    break;
  }
  case um2::MeshType::Quad: {
    *face_id = sp.getQuadMesh(cc.mesh_id).faceContaining(um2::Point2F(x, y));
    break;
  }
  case um2::MeshType::QuadraticTri: {
    *face_id = sp.getTri6Mesh(cc.mesh_id).faceContaining(um2::Point2F(x, y));
    break;
  }
  case um2::MeshType::QuadraticQuad: {
    *face_id = sp.getQuad8Mesh(cc.mesh_id).faceContaining(um2::Point2F(x, y));
    break;
  }
  default:
    LOG_ERROR("Invalid mesh type");
    return;
  }
}

void
um2MPACTCoarseCellFaceCentroid(void * const model, Int const cc_id, Int const face_id,
                               Float * const x, Float * const y)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const & cc = sp.getCoarseCell(cc_id);
  switch (cc.mesh_type) {
  case um2::MeshType::Tri: {
    auto const p = sp.getTriMesh(cc.mesh_id).getFace(face_id).centroid();
    *x = p[0];
    *y = p[1];
    break;
  }
  case um2::MeshType::Quad: {
    auto const p = sp.getQuadMesh(cc.mesh_id).getFace(face_id).centroid();
    *x = p[0];
    *y = p[1];
    break;
  }
  case um2::MeshType::QuadraticTri: {
    auto const p = sp.getTri6Mesh(cc.mesh_id).getFace(face_id).centroid();
    *x = p[0];
    *y = p[1];
    break;
  }
  case um2::MeshType::QuadraticQuad: {
    auto const p = sp.getQuad8Mesh(cc.mesh_id).getFace(face_id).centroid();
    *x = p[0];
    *y = p[1];
    break;
  }
  default:
    LOG_ERROR("Invalid mesh type");
    return;
  }
}

void
um2MPACTCoarseCellMaterialIDs(void * model, Int const cc_id, MatID * const mat_ids)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const & cc = sp.getCoarseCell(cc_id);
  for (Int i = 0; i < cc.numFaces(); ++i) {
    mat_ids[i] = cc.material_ids[i];
  }
}

void
um2MPACTIntersectCoarseCell(void * const model, Int const cc_id, Float const origin_x,
                            Float const origin_y, Float const direction_x,
                            Float const direction_y, Float * const intersections,
                            Int * const n)
{
  auto & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const & cc = sp.getCoarseCell(cc_id);
  um2::Ray2F const ray(um2::Point2F(origin_x, origin_y),
                       um2::Vec2F(direction_x, direction_y));

  switch (cc.mesh_type) {
  case um2::MeshType::Tri: {
    *n = sp.getTriMesh(cc.mesh_id).intersect(ray, intersections);
    break;
  }
  case um2::MeshType::Quad: {
    *n = sp.getQuadMesh(cc.mesh_id).intersect(ray, intersections);
    break;
  }
  case um2::MeshType::QuadraticTri: {
    *n = sp.getTri6Mesh(cc.mesh_id).intersect(ray, intersections);
    break;
  }
  case um2::MeshType::QuadraticQuad: {
    *n = sp.getQuad8Mesh(cc.mesh_id).intersect(ray, intersections);
    break;
  }
  default:
    LOG_ERROR("Invalid mesh type");
    *n = -1;
    return;
  }
  // Sort the intersections
  std::sort(intersections, intersections + *n);
}

void
um2MPACTRTMWidth(void * const model, Int const rtm_id, Float * const width)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *width = sp.getRTM(rtm_id).grid().extents(0);
}

void
um2MPACTRTMHeight(void * const model, Int const rtm_id, Float * const height)

{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  *height = sp.getRTM(rtm_id).grid().extents(1);
}

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
um2MPACTCoarseCellHeights(void * model, Int * const n, Int ** cc_ids, Float ** heights)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  um2::Vector<um2::Pair<Int, Float>> id_dz;
  auto constexpr eps = castIfNot<Float>(1e-4);
  // For each assembly
  //  For each lattice in the assembly
  //    Get the dz of the lattice
  //    For each rtm in the lattice
  //      For each coarse cell in the rtm
  //        If the id, dz pair is not in the vector, add it
  // Sort the vector by dz
  // For each unique assembly
  for (auto const & assembly : sp.assemblies()) {

    // For each lattice in the assembly
    Int const nlattices = assembly.children().size();
    for (Int ilat = 0; ilat < nlattices; ++ilat) {

      // Get the dz of the lattice
      Float const dz = assembly.grid().getBox(ilat).extents(0);

      // Get the lattice
      Int const lat_id = assembly.getChild(ilat);
      auto const & lattice = sp.getLattice(lat_id);

      // For each rtm in the lattice
      for (auto const & rtm_id : lattice.children()) {
        auto const & rtm = sp.getRTM(rtm_id);

        // For each coarse cell in the rtm
        for (auto const & cc_id : rtm.children()) {

          // If the id, dz pair is not in the vector, add it
          bool add_id = true;
          for (auto const & p : id_dz) {
            if (p.first == cc_id && um2::abs(p.second - dz) < eps) {
              add_id = false;
              break;
            }
          }
          if (add_id) {
            id_dz.emplace_back(cc_id, dz);
          }
        } // cc_id
      } // rtm_id
    } // ilat
  } // assembly

  // Sort the vector by dz first, then by id. But, if dz are close to each other,
  // then sort by id. This is to account for floating point errors.
  std::sort(id_dz.begin(), id_dz.end(), [eps](auto const & p1, auto const & p2) {
    return um2::abs(p1.second - p2.second) < eps ? p1.first < p2.first
                                                 : p1.second < p2.second;
  });
  *n = id_dz.size();
  size_t const n_bytes_int = static_cast<size_t>(*n) * sizeof(Int);
  size_t const n_bytes_float = static_cast<size_t>(*n) * sizeof(Float);
  *cc_ids = static_cast<Int *>(malloc(n_bytes_int));
  *heights = static_cast<Float *>(malloc(n_bytes_float));
  for (Int i = 0; i < id_dz.size(); ++i) {
    (*cc_ids)[i] = id_dz[i].first;
    (*heights)[i] = id_dz[i].second;
  }
}

void
um2MPACTRTMHeights(void * model, Int * const n, Int ** rtm_ids, Float ** heights)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  um2::Vector<um2::Pair<Int, Float>> id_dz;
  auto constexpr eps = castIfNot<Float>(1e-4);
  // For each assembly
  //  For each lattice in the assembly
  //    Get the dz of the lattice
  //    For each rtm in the lattice
  //      If the id, dz pair is not in the vector, add it
  // Sort the vector by dz

  // For each unique assembly
  for (auto const & assembly : sp.assemblies()) {

    // For each lattice in the assembly
    Int const nlattices = assembly.children().size();
    for (Int ilat = 0; ilat < nlattices; ++ilat) {

      // Get the dz of the lattice
      Float const dz = assembly.grid().getBox(ilat).extents(0);

      // Get the lattice
      Int const lat_id = assembly.getChild(ilat);
      auto const & lattice = sp.getLattice(lat_id);

      // For each rtm in the lattice
      for (auto const & rtm_id : lattice.children()) {
        // If the id, dz pair is not in the vector, add it
        bool add_id = true;
        for (auto const & p : id_dz) {
          if (p.first == rtm_id && um2::abs(p.second - dz) < eps) {
            add_id = false;
            break;
          }
        }
        if (add_id) {
          id_dz.emplace_back(rtm_id, dz);
        }
      } // rtm_id
    } // ilat
  } // assembly

  // Sort the vector by dz first, then by id. But, if dz are close to each other,
  // then sort by id. This is to account for floating point errors.
  std::sort(id_dz.begin(), id_dz.end(), [eps](auto const & p1, auto const & p2) {
    return um2::abs(p1.second - p2.second) < eps ? p1.first < p2.first
                                                 : p1.second < p2.second;
  });
  *n = id_dz.size();
  size_t const n_bytes_int = static_cast<size_t>(*n) * sizeof(Int);
  size_t const n_bytes_float = static_cast<size_t>(*n) * sizeof(Float);
  *rtm_ids = static_cast<Int *>(malloc(n_bytes_int));
  *heights = static_cast<Float *>(malloc(n_bytes_float));
  for (Int i = 0; i < id_dz.size(); ++i) {
    (*rtm_ids)[i] = id_dz[i].first;
    (*heights)[i] = id_dz[i].second;
  }
}

void
um2MPACTLatticeHeights(void * model, Int * const n, Int ** lat_ids, Float ** heights)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  um2::Vector<um2::Pair<Int, Float>> id_dz;
  auto constexpr eps = castIfNot<Float>(1e-4);
  // For each assembly
  //  For each lattice in the assembly
  //    Get the dz of the lattice
  //    For each rtm in the lattice
  //      If the id, dz pair is not in the vector, add it
  // Sort the vector by dz

  // For each unique assembly
  for (auto const & assembly : sp.assemblies()) {

    // For each lattice in the assembly
    Int const nlattices = assembly.children().size();
    for (Int ilat = 0; ilat < nlattices; ++ilat) {

      // Get the dz of the lattice
      Float const dz = assembly.grid().getBox(ilat).extents(0);

      // Get the lattice
      Int const lat_id = assembly.getChild(ilat);
      // If the id, dz pair is not in the vector, add it
      bool add_id = true;
      for (auto const & p : id_dz) {
        if (p.first == lat_id && um2::abs(p.second - dz) < eps) {
          add_id = false;
          break;
        }
      }
      if (add_id) {
        id_dz.emplace_back(lat_id, dz);
      }
    } // ilat
  } // assembly

  // Sort the vector by dz first, then by id. But, if dz are close to each other,
  // then sort by id. This is to account for floating point errors.
  std::sort(id_dz.begin(), id_dz.end(), [eps](auto const & p1, auto const & p2) {
    return um2::abs(p1.second - p2.second) < eps ? p1.first < p2.first
                                                 : p1.second < p2.second;
  });
  *n = id_dz.size();
  size_t const n_bytes_int = static_cast<size_t>(*n) * sizeof(Int);
  size_t const n_bytes_float = static_cast<size_t>(*n) * sizeof(Float);
  *lat_ids = static_cast<Int *>(malloc(n_bytes_int));
  *heights = static_cast<Float *>(malloc(n_bytes_float));
  for (Int i = 0; i < id_dz.size(); ++i) {
    (*lat_ids)[i] = id_dz[i].first;
    (*heights)[i] = id_dz[i].second;
  }
}

void
um2MPACTAssemblyHeights(void * const model, Int const asy_id, Float * const heights)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const & assembly = sp.getAssembly(asy_id);
  for (Int i = 0; i < assembly.children().size(); ++i) {
    heights[i] = assembly.grid().getBox(i).extents(0);
  }
}

void
um2MPACTCoarseCellFaceData(void * const model, Int const cc_id, Int * const mesh_type,
                           Int * const num_vertices, Int * const num_faces,
                           Float ** const vertices, Int ** const fv)
{
  auto const & sp = *reinterpret_cast<um2::mpact::Model *>(model);
  auto const & cc = sp.getCoarseCell(cc_id);
  auto const & mesh_id = cc.mesh_id;
  *mesh_type = static_cast<Int>(cc.mesh_type);
  switch (cc.mesh_type) {
  case um2::MeshType::Tri: {
    using Vertex = um2::TriFVM::Vertex;
    using FaceConn = um2::TriFVM::FaceConn;

    auto const & mesh = sp.getTriMesh(mesh_id);
    *num_vertices = mesh.numVertices();
    *num_faces = mesh.numFaces();

    size_t const n_bytes_float = static_cast<size_t>(*num_vertices) * sizeof(Vertex);
    *vertices = reinterpret_cast<Float *>(malloc(n_bytes_float));
    memcpy(*vertices, mesh.vertices().data(), n_bytes_float);

    size_t const n_bytes_int = static_cast<size_t>(*num_faces) * sizeof(FaceConn);
    *fv = reinterpret_cast<Int *>(malloc(n_bytes_int));
    memcpy(*fv, mesh.faceVertexConn().data(), n_bytes_int);
    break;
  }
  case um2::MeshType::Quad: {
    using Vertex = um2::QuadFVM::Vertex;
    using FaceConn = um2::QuadFVM::FaceConn;

    auto const & mesh = sp.getQuadMesh(mesh_id);
    *num_vertices = mesh.numVertices();
    *num_faces = mesh.numFaces();

    size_t const n_bytes_float = static_cast<size_t>(*num_vertices) * sizeof(Vertex);
    *vertices = reinterpret_cast<Float *>(malloc(n_bytes_float));
    memcpy(*vertices, mesh.vertices().data(), n_bytes_float);

    size_t const n_bytes_int = static_cast<size_t>(*num_faces) * sizeof(FaceConn);
    *fv = reinterpret_cast<Int *>(malloc(n_bytes_int));
    memcpy(*fv, mesh.faceVertexConn().data(), n_bytes_int);
    break;
  }
  case um2::MeshType::QuadraticTri: {
    using Vertex = um2::Tri6FVM::Vertex;
    using FaceConn = um2::Tri6FVM::FaceConn;

    auto const & mesh = sp.getTri6Mesh(mesh_id);
    *num_vertices = mesh.numVertices();
    *num_faces = mesh.numFaces();

    size_t const n_bytes_float = static_cast<size_t>(*num_vertices) * sizeof(Vertex);
    *vertices = reinterpret_cast<Float *>(malloc(n_bytes_float));
    memcpy(*vertices, mesh.vertices().data(), n_bytes_float);

    size_t const n_bytes_int = static_cast<size_t>(*num_faces) * sizeof(FaceConn);
    *fv = reinterpret_cast<Int *>(malloc(n_bytes_int));
    memcpy(*fv, mesh.faceVertexConn().data(), n_bytes_int);
    break;
  }
  case um2::MeshType::QuadraticQuad: {
    using Vertex = um2::Quad8FVM::Vertex;
    using FaceConn = um2::Quad8FVM::FaceConn;

    auto const & mesh = sp.getQuad8Mesh(mesh_id);
    *num_vertices = mesh.numVertices();
    *num_faces = mesh.numFaces();

    size_t const n_bytes_float = static_cast<size_t>(*num_vertices) * sizeof(Vertex);
    *vertices = reinterpret_cast<Float *>(malloc(n_bytes_float));
    memcpy(*vertices, mesh.vertices().data(), n_bytes_float);

    size_t const n_bytes_int = static_cast<size_t>(*num_faces) * sizeof(FaceConn);
    *fv = reinterpret_cast<Int *>(malloc(n_bytes_int));
    memcpy(*fv, mesh.faceVertexConn().data(), n_bytes_int);
    break;
  }
  default:
    LOG_ERROR("Invalid mesh type");
    return;
  }
}
