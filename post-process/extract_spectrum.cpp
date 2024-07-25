#include <um2.hpp>
#include <um2/stdlib/algorithm/is_sorted.hpp>

// NOLINTBEGIN(misc-include-cleaner)

PURE [[nodiscard]] auto
getSpectrum(um2::String const & filename) -> um2::Vector<Float>;

template <Int P, Int N>
PURE [[nodiscard]] auto
getSpectrum(um2::PolytopeSoup const & soup) -> um2::Vector<Float>;

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  // Get the file name from the command line
  if (argc != 2) {
    um2::String const exec_name(argv[0]);
    LOG_ERROR("Usage: ", exec_name, " <filename>");
    return 1;
  }

  // Get the spectrum
  um2::String const filename(argv[1]);
  auto const spectrum = getSpectrum(filename);

  // Write the spectrum to a file
  FILE * file = fopen("spectrum.txt", "w");
  if (file == nullptr) {
    LOG_ERROR("Could not open file spectrum.txt");
    return 1;
  }

  for (Int g = 0; g < spectrum.size(); ++g) {
    // Write group number and flux to file
    int const ret = fprintf(file, "%d, %.16f\n", g, spectrum[g]);
    if (ret < 0) {
      LOG_ERROR("Could not write to file spectrum.txt");
      return 1;
    }
  }
  int const ret = fclose(file);
  if (ret != 0) {
    LOG_ERROR("Could not close file spectrum.txt");
    return 1;
  }
  return 0;
}

PURE [[nodiscard]] auto
getSpectrum(um2::String const & filename) -> um2::Vector<Float>
{
  um2::PolytopeSoup const soup(filename);
  // For now, we assume that all the elements are the same type.
  auto const elem_types = soup.getElemTypes();
  if (elem_types.size() != 1) {
    LOG_ERROR("Expected only one element type, but found ", elem_types.size());
    return {};
  }
  switch (elem_types[0]) {
  case um2::VTKElemType::Triangle:
    LOG_INFO("FSR mesh type: Triangle");
    return getSpectrum<1, 3>(soup);
  case um2::VTKElemType::Quad:
    LOG_INFO("FSR mesh type: Quad");
    return getSpectrum<1, 4>(soup);
  case um2::VTKElemType::QuadraticTriangle:
    LOG_INFO("FSR mesh type: QuadraticTriangle");
    return getSpectrum<2, 6>(soup);
  case um2::VTKElemType::QuadraticQuad:
    LOG_INFO("FSR mesh type: QuadraticQuad");
    return getSpectrum<2, 8>(soup);
  default:
    LOG_ERROR("Unsupported element type");
    return {};
  }
}

template <Int P, Int N>
PURE [[nodiscard]] auto
getSpectrum(um2::PolytopeSoup const & soup) -> um2::Vector<Float>
{
  um2::FaceVertexMesh<P, N> const fvm(soup, /*validate=*/false);
  Int const num_faces = fvm.numFaces();

  // Get the area of each face
  um2::Vector<Float> face_areas(num_faces);
  for (Int i = 0; i < num_faces; ++i) {
    face_areas[i] = fvm.getFace(i).area();
  }

  um2::Vector<Float> spectrum;
  um2::Vector<Int> ids;
  um2::Vector<Float> flux;
  um2::String elset_name("flux_001");
  while (true) {
    flux.clear();
    soup.getElset(elset_name, ids, flux);
    if (flux.empty()) {
      break;
    }
    ASSERT(ids.size() == flux.size());
    ASSERT(ids.size() == num_faces);
    ASSERT(um2::is_sorted(ids.cbegin(), ids.cend()));
    Float group_flux = 0;
    Float local_accum = 0;
    for (Int i = 0; i < ids.size(); ++i) {
      local_accum += flux[i] * face_areas[i];
      // Increment group flux every 128 faces
      if (i % 128 == 127) {
        group_flux += local_accum;
        local_accum = 0;
      }
    }
    group_flux += local_accum;
    spectrum.emplace_back(group_flux);
    um2::mpact::incrementASCIINumber(elset_name);
  }
  return spectrum;
}

// NOLINTEND(misc-include-cleaner)
