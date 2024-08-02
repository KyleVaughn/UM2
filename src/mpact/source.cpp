#include <um2/common/logger.hpp>
#include <um2/config.hpp>
#include <um2/math/vec.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/mesh/polytope_soup.hpp>
#include <um2/mpact/model.hpp>
#include <um2/mpact/source.hpp>
#include <um2/physics/material.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/vector.hpp>

namespace um2::mpact
{

namespace
{

// Specialize for each mesh type
template <Int P, Int N>
[[nodiscard]] auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
getSource(PolytopeSoup const & soup, Vector<Material> const & materials) -> Vector<Vec2F>
{
  //
  // Q_i = A_i \sum_g (\Sigma_{s,g} + \frac{1}{k} \nu \Sigma_{f,g}) \phi_g
  //
  // We can't obtain k from the FSR output, so we omit k and return the
  // scattering and fission source separately.

  FaceVertexMesh<P, N> const fvm(soup, /*validate=*/false);
  Int const num_faces = fvm.numFaces();

  // Get the material ID of each face
  Vector<Int> ids(num_faces);
  Vector<Int> mat_ids(num_faces);
  {
    Vector<Float> mat_data;
    soup.getElset("material", ids, mat_data);
    ASSERT(ids.size() == num_faces);
    ASSERT(mat_data.size() == num_faces);
    // Convert each Float ID to Int and -1 for 1-based indexing
    for (Int i = 0; i < num_faces; ++i) {
      mat_ids[i] = static_cast<Int>(mat_data[i]) - 1;
    }
  }

  // For each energy group, calculate the source
  //---------------------------------------------------------------------------
  Vector<Vec2F> source(num_faces);
  // Explicitly zero the source
  for (auto & s : source) {
    s[0] = 0;
    s[1] = 0;
  }

  Vector<Float> flux(num_faces);
  String elset_name = "flux_001";
  Int const ngroups = materials[0].xsec().numGroups();
  for (Int ig = 0; ig < ngroups; ++ig) {
    // Get the flux data for group ig
    soup.getElset(elset_name, ids, flux);
    ASSERT(ids.size() == num_faces);
    ASSERT(flux.size() == num_faces);
    // For each face, reduce onto the source
    for (Int i = 0; i < num_faces; ++i) {
      // Get the material
      auto const & material = materials[mat_ids[i]];

      // Get the scattering and nu-fission cross sections
      auto const sigma_s = material.xsec().s()[ig];
      auto const nu_sigma_f = material.xsec().nuf()[ig];

      // Calculate the source
      source[i][0] += flux[i] * sigma_s;
      source[i][1] += flux[i] * nu_sigma_f;
    }

    // Increment the elset name
    um2::mpact::incrementASCIINumber(elset_name);
  }

  // Multiply by the area of the face
  // for (Int i = 0; i < num_faces; ++i) {
  //  source[i] *= fvm.getFace(i).area();
  //}

  return source;
}

} // namespace

PURE [[nodiscard]] auto
getSource(PolytopeSoup const & soup, Vector<Material> const & materials) -> Vector<Vec2F>
{
  LOG_INFO("Extracting source from MPACT FSR output");

  // For now, we assume that all the elements are the same type.
  auto const elem_types = soup.getElemTypes();
  if (elem_types.size() != 1) {
    LOG_ERROR("Expected only one element type, but found ", elem_types.size());
    return {};
  }
  switch (elem_types[0]) {
  case VTKElemType::Triangle:
    LOG_INFO("FSR mesh type: Triangle");
    return getSource<1, 3>(soup, materials);
  case VTKElemType::Quad:
    LOG_INFO("FSR mesh type: Quad");
    return getSource<1, 4>(soup, materials);
  case VTKElemType::QuadraticTriangle:
    LOG_INFO("FSR mesh type: QuadraticTriangle");
    return getSource<2, 6>(soup, materials);
  case VTKElemType::QuadraticQuad:
    LOG_INFO("FSR mesh type: QuadraticQuad");
    return getSource<2, 8>(soup, materials);
  default:
    LOG_ERROR("Unsupported element type");
    return {};
  }
}

} // namespace um2::mpact
