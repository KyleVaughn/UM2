#include <um2c.h>

#define TRY_CATCH(func)                                                                  \
  if (ierr != nullptr) {                                                                 \
    *ierr = 0;                                                                           \
  }                                                                                      \
  try {                                                                                  \
    func;                                                                                \
  } catch (...) {                                                                        \
    if (ierr != nullptr) {                                                               \
      *ierr = 1;                                                                         \
    }                                                                                    \
  }

//==============================================================================
// um2Malloc and um2Free
//==============================================================================

void
um2Malloc(void ** const p, Size const size)
{
  *p = malloc(static_cast<size_t>(size));
}

void
um2Free(void * const p)
{
  if (p != nullptr) {
    free(p);
  }
}

//==============================================================================
// um2Initialize and um2Finalize
//==============================================================================

void
um2Initialize(char const * const verbosity, Int const init_gmsh, Int const gmsh_verbosity,
              Int * const ierr)
{
  TRY_CATCH(um2::initialize(verbosity, init_gmsh == 1, gmsh_verbosity));
}

void
um2Finalize(Int * const ierr)
{
  TRY_CATCH(um2::finalize());
}

void
getSizeOfInt(int * size)
{
  *size = static_cast<int>(sizeof(Int));
}

void
getSizeOfFloat(int * size)
{
  *size = static_cast<int>(sizeof(Float));
}

//==============================================================================
// MPACT SpatialPartition
//==============================================================================

void
um2NewMPACTSpatialPartition(void ** const model, Int * const ierr)
{
  TRY_CATCH(*model = reinterpret_cast<void *>(new um2::mpact::SpatialPartition()));
}

void
um2DeleteMPACTSpatialPartition(void * const model, Int * const ierr)
{
  TRY_CATCH(delete reinterpret_cast<um2::mpact::SpatialPartition *>(model));
}

void
um2ImportMPACTModel(char const * const path, void ** const model, Int * const ierr)
{
  TRY_CATCH({
    std::string const path_str(path);
    *model = reinterpret_cast<void *>(new um2::mpact::SpatialPartition());
    um2::importMesh(path_str, *reinterpret_cast<um2::mpact::SpatialPartition *>(*model));
  });
}

void
um2MPACTNumCoarseCells(void * const model, Int * const n, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.coarse_cells.size();
  });
}

void
um2MPACTNumRTMs(void * const model, Int * const n, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.rtms.size();
  });
}

void
um2MPACTNumLattices(void * const model, Int * const n, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.lattices.size();
  });
}

void
um2MPACTNumAssemblies(void * const model, Int * const n, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.assemblies.size();
  });
}

void
um2MPACTCoreNumCells(void * const model, Int * const nx, Int * const ny, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    auto const ncells = sp.core.numCells();
    *nx = ncells[0];
    *ny = ncells[1];
  });
}

void
um2MPACTAssemblyNumCells(void * const model, Int const asy_id, Int * const nx,
                         Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *nx = sp.assemblies[asy_id].numXCells();
  });
}

void
um2MPACTLatticeNumCells(void * const model, Int const lat_id, Int * const nx,
                        Int * const ny, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    auto const ncells = sp.lattices[lat_id].numCells();
    *nx = ncells[0];
    *ny = ncells[1];
  });
}

void
um2MPACTRTMNumCells(void * const model, Int const rtm_id, Int * const nx, Int * const ny,
                    Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    auto const ncells = sp.rtms[rtm_id].numCells();
    *nx = ncells[0];
    *ny = ncells[1];
  });
}

void
um2MPACTCoreGetChild(void * const model, Int const ix, Int const iy, Int * const child,
                     Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *child = sp.core.getChild(ix, iy);
  });
}

void
um2MPACTAssemblyGetChild(void * const model, Int const asy_id, Int const ix,
                         Int * const child, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *child = sp.assemblies[asy_id].getChild(ix);
  });
}

void
um2MPACTLatticeGetChild(void * const model, Int const lat_id, Int const ix, Int const iy,
                        Int * const child, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *child = sp.lattices[lat_id].getChild(ix, iy);
  });
}

void
um2MPACTRTMGetChild(void * const model, Int const rtm_id, Int const ix, Int const iy,
                    Int * const child, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *child = sp.rtms[rtm_id].getChild(ix, iy);
  });
}

//==============================================================================
// CoarseCell
//==============================================================================

void
um2MPACTCoarseCellNumFaces(void * const model, Int const cc_id, Int * const num_faces,
                           Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *num_faces = sp.coarse_cells[cc_id].numFaces();
  });
}

void
um2MPACTCoarseCellWidth(void * const model, Int const cc_id, Float * const width,
                        Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *width = sp.coarse_cells[cc_id].dxdy[0];
  });
}

void
um2MPACTCoarseCellHeight(void * const model, Int const cc_id, Float * const height,
                         Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *height = sp.coarse_cells[cc_id].dxdy[1];
  });
}

void
um2MPACTCoarseCellFaceAreas(void * const model, Int const cc_id, Int * const n,
                            Float ** const areas, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    auto const & cc = sp.coarse_cells[cc_id];
    auto const & mesh_id = cc.mesh_id;
    um2::Vector<Float> areas_vec;
    // NOLINTBEGIN(bugprone-branch-clone)
    switch (cc.mesh_type) {
    case um2::MeshType::Tri: {
      sp.tri[mesh_id].getFaceAreas(areas_vec);
      break;
    }
    case um2::MeshType::Quad: {
      sp.quad[mesh_id].getFaceAreas(areas_vec);
      break;
    }
    case um2::MeshType::QuadraticTri: {
      sp.quadratic_tri[mesh_id].getFaceAreas(areas_vec);
      break;
    }
    case um2::MeshType::QuadraticQuad: {
      sp.quadratic_quad[mesh_id].getFaceAreas(areas_vec);
      break;
    }
    default:
      um2::Log::error("Mesh type not supported");
      // NOLINTNEXTLINE justification: complains this is a null deference, but it's not
      *ierr = 1;
      break;
    }
    // NOLINTEND(bugprone-branch-clone)
    *n = areas_vec.size();
    *areas = static_cast<Float *>(malloc(static_cast<size_t>(*n) * sizeof(Float)));
    std::copy(areas_vec.begin(), areas_vec.end(), *areas);
  });
}

void
um2MPACTCoarseCellFaceContaining(void * const model, Int const cc_id, Float const x,
                                 Float const y, Int * const face_id, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    auto const & cc = sp.coarse_cells[cc_id];
    auto const & mesh_id = cc.mesh_id;
    // NOLINTBEGIN(bugprone-branch-clone)
    switch (cc.mesh_type) {
    case um2::MeshType::Tri: {
      *face_id = sp.tri[mesh_id].faceContaining(um2::Point2<Float>(x, y));
      break;
    }
    case um2::MeshType::Quad: {
      *face_id = sp.quad[mesh_id].faceContaining(um2::Point2<Float>(x, y));
      break;
    }
    case um2::MeshType::QuadraticTri: {
      *face_id = sp.quadratic_tri[mesh_id].faceContaining(um2::Point2<Float>(x, y));
      break;
    }
    case um2::MeshType::QuadraticQuad: {
      *face_id = sp.quadratic_quad[mesh_id].faceContaining(um2::Point2<Float>(x, y));
      break;
    }
    default:
      um2::Log::error("Mesh type not supported");
      // NOLINTNEXTLINE justification: complains this is a null deference, but it's not
      *ierr = 1;
      break;
    }
    // NOLINTEND(bugprone-branch-clone)
  });
}

void
um2MPACTCoarseCellFaceCentroid(void * const model, Int const cc_id, Int const face_id,
                               Float * const x, Float * const y, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    auto const & cc = sp.coarse_cells[cc_id];
    auto const & mesh_id = cc.mesh_id;
    // NOLINTBEGIN(bugprone-branch-clone)
    switch (cc.mesh_type) {
    case um2::MeshType::Tri: {
      auto const p = sp.tri[mesh_id].getFace(face_id).centroid();
      *x = p[0];
      *y = p[1];
      break;
    }
    case um2::MeshType::Quad: {
      auto const p = sp.quad[mesh_id].getFace(face_id).centroid();
      *x = p[0];
      *y = p[1];
      break;
    }
    case um2::MeshType::QuadraticTri: {
      auto const p = sp.quadratic_tri[mesh_id].getFace(face_id).centroid();
      *x = p[0];
      *y = p[1];
      break;
    }
    case um2::MeshType::QuadraticQuad: {
      auto const p = sp.quadratic_quad[mesh_id].getFace(face_id).centroid();
      *x = p[0];
      *y = p[1];
      break;
    }
    default:
      um2::Log::error("Mesh type not supported");
      // NOLINTNEXTLINE justification: complains this is a null deference, but it's not
      *ierr = 1;
      break;
    }
    // NOLINTEND(bugprone-branch-clone)
  });
}

void
um2MPACTCoarseCellMaterialIDs(void * model, Int cc_id, MaterialID ** mat_ids, Int * n,
                              Int * ierr)
{
  TRY_CATCH({
    auto & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    auto & cc = sp.coarse_cells[cc_id];
    *n = cc.material_ids.size();
    *mat_ids = cc.material_ids.begin();
  });
}

// NOLINTBEGIN
void
um2MPACTIntersectCoarseCell(void * const model, Int const cc_id, Float const origin_x,
                            Float const origin_y, Float const direction_x,
                            Float const direction_y, Float * const intersections,
                            Int * const n, Int * const ierr)
{
  *ierr = 1;
  if (*n <= 0) {
    return;
  }
  auto & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
  auto const & cc = sp.coarse_cells[cc_id];
  auto const & mesh_id = cc.mesh_id;
  um2::Ray<2, Float> const ray(um2::Point<2, Float>(origin_x, origin_y),
                               um2::Vec<2, Float>(direction_x, direction_y));

  switch (cc.mesh_type) {
  case um2::MeshType::Tri: {
    um2::intersect(sp.tri[mesh_id], ray, intersections, n);
    break;
  }
  case um2::MeshType::Quad: {
    um2::intersect(sp.quad[mesh_id], ray, intersections, n);
    break;
  }
  case um2::MeshType::QuadraticTri: {
    um2::intersect(sp.quadratic_tri[mesh_id], ray, intersections, n);
    break;
  }
  case um2::MeshType::QuadraticQuad: {
    um2::intersect(sp.quadratic_quad[mesh_id], ray, intersections, n);
    break;
  }
  default:
    um2::Log::error("Mesh type not supported");
    break;
  }
  *ierr = 0;
}
// NOLINTEND

void
um2MPACTRTMWidth(void * const model, Int const rtm_id, Float * const width,
                 Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *width = sp.rtms[rtm_id].boundingBox().width();
  });
}

void
um2MPACTRTMHeight(void * const model, Int const rtm_id, Float * const height,
                  Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *height = sp.rtms[rtm_id].boundingBox().height();
  });
}

void
// NOLINTBEGIN
um2MPACTCoarseCellHeights(void * model, Int * const n, Int ** const cc_ids,
                          Float ** heights, Int * const ierr)
{
  *ierr = 1;
  auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
  std::vector<std::pair<Int, Float>> id_dz;
#if UM2_ENABLE_FLOAT64 == 0
  auto const eps = static_cast<Float>(1e-4);
#else
  auto const eps = 1e-4;
#endif
  // For each assembly
  //  For each lattice in the assembly
  //    Get the dz of the lattice
  //    For each rtm in the lattice
  //      For each coarse cell in the rtm
  //        If the id, dz pair is not in the vector, add it
  // Sort the vector by dz
  // For each unique assembly
  for (auto const & assembly : sp.assemblies) {
    Size const nlattices = assembly.children.size();
    // For each lattice in the assembly
    for (Size ilat = 0; ilat < nlattices; ++ilat) {
      Int const lat_id = assembly.children[ilat];
      // Get the dz of the lattice
      auto const bb = assembly.getBox(ilat);
      Float const dz = bb.width();
      auto const & lattice = sp.lattices[lat_id];
      // For each rtm in the lattice
      for (auto const & rtm_id : lattice.children) {
        auto const & rtm = sp.rtms[rtm_id];
        // For each coarse cell in the rtm
        for (auto const & cc_id : rtm.children) {
          // If the id, dz pair is not in the vector, add it
          bool const add_id =
              !std::any_of(id_dz.begin(), id_dz.end(), [cc_id, dz, eps](auto const & p) {
                return p.first == cc_id && std::abs(p.second - dz) < eps;
              });
          if (add_id) {
            id_dz.emplace_back(cc_id, dz);
          }
        } // icc
      }   // irtm
    }     // ilat
  }       // assembly
  // Sort the vector by dz first, then by id. But, if dz are close to each other,
  // then sort by id.
  std::sort(id_dz.begin(), id_dz.end(), [eps](auto const & p1, auto const & p2) {
    return std::abs(p1.second - p2.second) < eps ? p1.first < p2.first
                                                 : p1.second < p2.second;
  });
  *n = static_cast<Int>(id_dz.size());
  *cc_ids = static_cast<Int *>(malloc(id_dz.size() * sizeof(Int)));
  *heights = static_cast<Float *>(malloc(id_dz.size() * sizeof(Float)));
  for (size_t i = 0; i < id_dz.size(); ++i) {
    (*cc_ids)[i] = id_dz[i].first;
    (*heights)[i] = id_dz[i].second;
  }
  *ierr = 0;
}
// NOLINTEND

void
// NOLINTBEGIN
um2MPACTRTMHeights(void * model, Int * const n, Int ** const rtm_ids, Float ** heights,
                   Int * const ierr)
{
  *ierr = 1;
  auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
  std::vector<std::pair<Int, Float>> id_dz;
#if UM2_ENABLE_FLOAT64 == 0
  auto const eps = static_cast<Float>(1e-4);
#else
  auto const eps = 1e-4;
#endif
  // For each assembly
  //  For each lattice in the assembly
  //    Get the dz of the lattice
  //    For each rtm in the lattice
  //      If the id, dz pair is not in the vector, add it
  // Sort the vector by dz
  // For each unique assembly
  for (auto const & assembly : sp.assemblies) {
    Size const nlattices = assembly.children.size();
    // For each lattice in the assembly
    for (Size ilat = 0; ilat < nlattices; ++ilat) {
      Int const lat_id = assembly.children[ilat];
      // Get the dz of the lattice
      auto const bb = assembly.getBox(ilat);
      Float const dz = bb.width();
      auto const & lattice = sp.lattices[lat_id];
      // For each rtm in the lattice
      for (auto const & rtm_id : lattice.children) {
        // If the id, dz pair is not in the vector, add it
        bool const add_id =
            !std::any_of(id_dz.begin(), id_dz.end(), [rtm_id, dz, eps](auto const & p) {
              return p.first == rtm_id && std::abs(p.second - dz) < eps;
            });
        if (add_id) {
          id_dz.emplace_back(rtm_id, dz);
        }
      } // irtm
    }   // ilat
  }     // assembly
  // Sort the vector by dz first, then by id. But, if dz are close to each other,
  // then sort by id.
  std::sort(id_dz.begin(), id_dz.end(), [eps](auto const & p1, auto const & p2) {
    return std::abs(p1.second - p2.second) < eps ? p1.first < p2.first
                                                 : p1.second < p2.second;
  });
  *n = static_cast<Int>(id_dz.size());
  *rtm_ids = static_cast<Int *>(malloc(id_dz.size() * sizeof(Int)));
  *heights = static_cast<Float *>(malloc(id_dz.size() * sizeof(Float)));
  for (size_t i = 0; i < id_dz.size(); ++i) {
    (*rtm_ids)[i] = id_dz[i].first;
    (*heights)[i] = id_dz[i].second;
  }
  *ierr = 0;
}
// NOLINTEND

void
// NOLINTBEGIN
um2MPACTLatticeHeights(void * model, Int * const n, Int ** const lat_ids,
                       Float ** heights, Int * const ierr)
{
  *ierr = 1;
  auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
  std::vector<std::pair<Int, Float>> id_dz;
#if UM2_ENABLE_FLOAT64 == 0
  auto const eps = static_cast<Float>(1e-4);
#else
  auto const eps = 1e-4;
#endif
  // For each assembly
  //  For each lattice in the assembly
  //    Get the dz of the lattice
  //    For each rtm in the lattice
  //      If the id, dz pair is not in the vector, add it
  // Sort the vector by dz
  // For each unique assembly
  for (auto const & assembly : sp.assemblies) {
    Size const nlattices = assembly.children.size();
    // For each lattice in the assembly
    for (Size ilat = 0; ilat < nlattices; ++ilat) {
      Int const lat_id = assembly.children[ilat];
      // Get the dz of the lattice
      auto const bb = assembly.getBox(ilat);
      Float const dz = bb.width();
      // If the id, dz pair is not in the vector, add it
      bool const add_id =
          !std::any_of(id_dz.begin(), id_dz.end(), [lat_id, dz, eps](auto const & p) {
            return p.first == lat_id && std::abs(p.second - dz) < eps;
          });
      if (add_id) {
        id_dz.emplace_back(lat_id, dz);
      }
    } // ilat
  }   // assembly
  // Sort the vector by dz first, then by id. But, if dz are close to each other,
  // then sort by id.
  std::sort(id_dz.begin(), id_dz.end(), [eps](auto const & p1, auto const & p2) {
    return std::abs(p1.second - p2.second) < eps ? p1.first < p2.first
                                                 : p1.second < p2.second;
  });
  *n = static_cast<Int>(id_dz.size());
  *lat_ids = static_cast<Int *>(malloc(id_dz.size() * sizeof(Int)));
  *heights = static_cast<Float *>(malloc(id_dz.size() * sizeof(Float)));
  for (size_t i = 0; i < id_dz.size(); ++i) {
    (*lat_ids)[i] = id_dz[i].first;
    (*heights)[i] = id_dz[i].second;
  }
  *ierr = 0;
}
// NOLINTEND

void
um2MPACTAssemblyHeights(void * const model, Int const asy_id, Int * const n,
                        Float ** const heights, Int * const ierr)
{
  TRY_CATCH({
    auto const & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
    *n = sp.assemblies[asy_id].children.size();
    *heights = static_cast<Float *>(malloc(static_cast<size_t>(*n) * sizeof(Float)));
    for (Int i = 0; i < *n; ++i) {
      (*heights)[i] = sp.assemblies[asy_id].getBox(i).width();
    }
  });
}

void
um2MPACTCoarseCellFaceData(void * const model, Int const cc_id, Int * const mesh_type,
                           Int * const num_vertices, Int * const num_faces,
                           Float ** const vertices, Int ** const fv, Int * const ierr)
{
  auto & sp = *reinterpret_cast<um2::mpact::SpatialPartition *>(model);
  auto const & cc = sp.coarse_cells[cc_id];
  auto const & mesh_id = cc.mesh_id;
  *mesh_type = static_cast<Int>(cc.mesh_type);
  switch (cc.mesh_type) {
  case um2::MeshType::Tri: {
    *num_vertices = sp.tri[mesh_id].numVertices();
    *num_faces = sp.tri[mesh_id].numFaces();
    *vertices = reinterpret_cast<Float *>(sp.tri[mesh_id].vertices.data());
    *fv = reinterpret_cast<Int *>(sp.tri[mesh_id].fv.data());
    break;
  }
  case um2::MeshType::Quad: {
    *num_vertices = sp.quad[mesh_id].numVertices();
    *num_faces = sp.quad[mesh_id].numFaces();
    *vertices = reinterpret_cast<Float *>(sp.quad[mesh_id].vertices.data());
    *fv = reinterpret_cast<Int *>(sp.quad[mesh_id].fv.data());
    break;
  }
  case um2::MeshType::QuadraticTri: {
    *num_vertices = sp.quadratic_tri[mesh_id].numVertices();
    *num_faces = sp.quadratic_tri[mesh_id].numFaces();
    *vertices = reinterpret_cast<Float *>(sp.quadratic_tri[mesh_id].vertices.data());
    *fv = reinterpret_cast<Int *>(sp.quadratic_tri[mesh_id].fv.data());
    break;
  }
  case um2::MeshType::QuadraticQuad: {
    *num_vertices = sp.quadratic_quad[mesh_id].numVertices();
    *num_faces = sp.quadratic_quad[mesh_id].numFaces();
    *vertices = reinterpret_cast<Float *>(sp.quadratic_quad[mesh_id].vertices.data());
    *fv = reinterpret_cast<Int *>(sp.quadratic_quad[mesh_id].fv.data());
    break;
  }
  default: {
    um2::Log::error("Mesh type not supported");
    *ierr = 1;
    return;
  }
  }
  *ierr = 0;
}
