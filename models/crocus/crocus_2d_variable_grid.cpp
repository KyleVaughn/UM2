// Model reference:
//  Paratte, J. M., et al. "A benchmark on the calculation of kinetic parameters based
//  on reactivity effect experiments in the CROCUS reactor." Annals of Nuclear energy
//  33.8 (2006): 739-748.
//  https://doi.org/10.1016/j.anucene.2005.09.012

// NOTE: We omit small gas gaps for more clad material in order to prevent
//       mesh generation issues.

// NOLINTBEGIN(misc-include-cleaner)

#include <um2.hpp>

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  //===========================================================================
  // Parse command line arguments
  //===========================================================================

  // Check the number of arguments
  if (argc != 2) {
    um2::String const exec_name(argv[0]);
    um2::logger::error("Usage: ", exec_name, " num_coarse_cells");
    return 1;
  }

  char * end = nullptr;
  Int const num_coarse_cells = um2::strto<Int>(argv[1], &end);
  ASSERT(end != nullptr);
  ASSERT(num_coarse_cells > 0);

  //============================================================================
  // Materials
  //============================================================================
  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" +
                             um2::mpact::XSLIB_51G);

  // NOTE: number densities should be computed from the source, but I have simply
  // ripped them from another CROCUS model for now.

  Float constexpr temp = 293.15; // K. pg. 744 Sec. 3.1

  // UO2
  //---------------------------------------------------------------------------
  um2::Material uo2;
  uo2.setName("UO2");
  uo2.setDensity(10.556); // pg. 742 Sec. 2.3
  uo2.setTemperature(temp);
  uo2.setColor(um2::orange); // Match Fig. 4
  uo2.addNuclide(92235, 4.30565e-04);
  uo2.addNuclide(92238, 2.31145e-02);
  uo2.addNuclide(8016, 4.70902e-02);
  uo2.populateXSec(xslib);

  // Clad
  //---------------------------------------------------------------------------
  um2::Material clad;
  clad.setName("Clad");
  clad.setDensity(2.70); // pg. 743 Table 1
  clad.setTemperature(temp);
  clad.setColor(um2::slategray);
  clad.addNuclide(13027, 6.02611e-02);
  clad.populateXSec(xslib);

  // Umetal
  //---------------------------------------------------------------------------
  um2::Material umetal;
  umetal.setName("Umetal");
  umetal.setDensity(18.677); // pg. 742 Sec. 2.3
  umetal.setTemperature(temp);
  umetal.setColor(um2::red);
  umetal.addNuclide(92235, 4.53160e-04);
  umetal.addNuclide(92238, 4.68003e-02);
  umetal.populateXSec(xslib);

  // Water
  //---------------------------------------------------------------------------
  um2::Material water;
  water.setName("Water");
  water.setDensity(0.9983); // pg. 743 Table 1
  water.setTemperature(temp);
  water.setColor(um2::blue);
  water.addNuclide(1001, 6.67578e-02);
  water.addNuclide(8016, 3.33789e-02);
  water.populateXSec(xslib);

  //============================================================================
  // Geometry
  //============================================================================

  // Given Parameters
  //---------------------------------------------------------------------------
  Float const d_uo2_fuel = 1.052;    // UO2 fuel diameter pg. 741 Fig. 2
  Float const d_uo2_clad = 1.260;    // UO2 clad diameter pg. 741 Fig. 2
  Float const uo2_pitch = 1.837;     // UO2 pin pitch pg. 742 Sec. 2.3
  Float const d_umetal_fuel = 1.700; // Umetal fuel diameter pg. 741 Fig. 2
  Float const d_umetal_clad = 1.935; // Umetal clad diameter pg. 741 Fig. 2
  Float const umetal_pitch = 2.917;  // Umetal pin pitch pg. 742 Sec. 2.3

  // Computed parameters
  //---------------------------------------------------------------------------
  Float const r_uo2_fuel = d_uo2_fuel / 2;
  Float const r_uo2_clad = d_uo2_clad / 2;
  Float const r_umetal_fuel = d_umetal_fuel / 2;
  Float const r_umetal_clad = d_umetal_clad / 2;

  um2::Vector<Float> const uo2_radii = {r_uo2_fuel, r_uo2_clad};
  um2::Vector<Float> const umetal_radii = {r_umetal_fuel, r_umetal_clad};

  um2::Vector<um2::Material> const uo2_mats = {uo2, clad};
  um2::Vector<um2::Material> const umetal_mats = {umetal, clad};

  um2::Vector<um2::Vector<Int>> const uo2_pin_lattice = um2::stringToLattice<Int>(R"(
    0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
  )");

  um2::Vector<um2::Vector<Int>> const umetal_pin_lattice = um2::stringToLattice<Int>(R"(
    0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0
    0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 1 0 0
    0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 0
    0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0
    1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0
    1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
    1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
    0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
    0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0
    0 0 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0
    0 0 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 1 0 0
    0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0)");

  // Depending on how much water we want to simulate, we can change where the center of
  // the problem is located. x_center == y_center == center
  Float const center = 20 * umetal_pitch;

  // Compute the offset of the UO2 and Umetal lattices from the center of the problem
  auto const n_uo2 = static_cast<Float>(uo2_pin_lattice.size());
  auto const n_umetal = static_cast<Float>(umetal_pin_lattice.size());
  Float const uo2_offset = center - 0.5 * n_uo2 * uo2_pitch;
  Float const umetal_offset = center - 0.5 * n_umetal * umetal_pitch;

  um2::Vector<um2::Vec2d> const uo2_dxdy(2, {uo2_pitch, uo2_pitch});
  um2::gmsh::model::occ::addCylindricalPinLattice2D(
      uo2_pin_lattice,         // Pin IDs
      uo2_dxdy,                // Pitch of the pins
      {{}, uo2_radii},         // Radii of the pins
      {{}, uo2_mats},          // Materials of the pins
      {uo2_offset, uo2_offset} // Offset of the lattice
  );

  um2::Vector<um2::Vec2d> const umetal_dxdy(2, {umetal_pitch, umetal_pitch});
  um2::gmsh::model::occ::addCylindricalPinLattice2D(
      umetal_pin_lattice,            // Pin IDs
      umetal_dxdy,                   // Pitch of the pins
      {{}, umetal_radii},            // Radii of the pins
      {{}, umetal_mats},             // Materials of the pins
      {umetal_offset, umetal_offset} // Offset of the lattice
  );

  //===========================================================================
  // Overlay CMFD mesh
  //===========================================================================

  // Construct the MPACT model
  um2::mpact::Model model;
  model.addMaterial(uo2);
  model.addMaterial(clad);
  model.addMaterial(umetal);
  model.addMaterial(water);

  // Add a coarse grid that evenly subdivides the domain
  um2::Vec2F const domain_extents(2 * center, 2 * center);
  um2::Vec2I const num_cells(num_coarse_cells, num_coarse_cells);
  model.addCoarseGrid(domain_extents, num_cells);
  um2::gmsh::model::occ::overlayCoarseGrid(model, water);

  //===========================================================================
  // Generate the mesh
  //===========================================================================

  um2::gmsh::model::mesh::setGlobalMeshSize(uo2_pitch / 8);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::QuadraticTri);
  um2::gmsh::write("crocus_2d.inp");

  //===========================================================================
  // Complete the MPACT model and write the mesh
  //===========================================================================

  model.importCoarseCellMeshes("crocus_2d.inp");
  // model.writeCMFDInfo("crocus_2d_cmfd_info.xdmf");
  model.write("crocus_2d.xdmf");
  um2::finalize();
  return 0;
}

// NOLINTEND(misc-include-cleaner)
