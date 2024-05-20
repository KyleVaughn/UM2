// Model reference:
//  VERA Core Physics Benchmark Progression Problem Specifications
//  Revision 4, August 29, 2014
//  CASL-U-2012-0131-004

#include <um2.hpp>

auto
main(int argc, char** argv) -> int
{
  um2::initialize();

  // Check the number of arguments
  if (argc != 2) {
    um2::logger::error("Usage: ./4_2d num_coarse_cells");
    return 1;
  }

  //===========================================================================
  // Parametric study parameters
  //===========================================================================

  char * end = nullptr;
  Int const num_coarse_cells = um2::strto<Int>(argv[1], &end);
  ASSERT(end != nullptr);
  ASSERT(num_coarse_cells > 0);

  //============================================================================
  // Materials
  //============================================================================
  // Nuclides and number densities from table P4-3 (pg. 54)

  um2::XSLibrary const xslib(um2::settings::xs::library_path + "/" + um2::mpact::XSLIB_51G);

  // Fuel_2.110%
  //---------------------------------------------------------------------------
  um2::Material fuel_2110;
  fuel_2110.setName("Fuel_2.11%");
  fuel_2110.setDensity(10.257); // g/cm^3, Table P4-1 (pg. 50)
  fuel_2110.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  fuel_2110.setColor(um2::red); // Match Fig. P4-2 (pg. 53)
  // Number densities in atoms/b-cm from Table P4-3 (pg. 54)
  fuel_2110.addNuclide("O16", 4.57591e-02);
  fuel_2110.addNuclide("U234", 4.04814e-06);
  fuel_2110.addNuclide("U235", 4.88801e-04);
  fuel_2110.addNuclide("U236", 2.23756e-06);
  fuel_2110.addNuclide("U238", 2.23844e-02);
  fuel_2110.populateXSec(xslib);

  // Fuel_2.619%
  //---------------------------------------------------------------------------
  um2::Material fuel_2619;
  fuel_2619.setName("Fuel_2.619%");
  fuel_2619.setDensity(10.257); // g/cm^3, Table P4-1 (pg. 50)
  fuel_2619.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  fuel_2619.setColor(um2::green); // Match Fig. P4-2 (pg. 53)
  // Number densities in atoms/b-cm from Table P4-3 (pg. 54)
  fuel_2619.addNuclide("O16", 4.57617e-02);
  fuel_2619.addNuclide("U234", 5.09503e-06);
  fuel_2619.addNuclide("U235", 6.06733e-04);
  fuel_2619.addNuclide("U236", 2.76809e-06);
  fuel_2619.addNuclide("U238", 2.22663e-02);
  fuel_2619.populateXSec(xslib);

  // Gap
  //---------------------------------------------------------------------------
  um2::Material gap;
  gap.setName("Gap");
  // "Helium with nominal density" (pg. 22). Helium at 565 K and 2250 psia, according to
  // NIST has a density of 0.012768 g/cm^3.
  gap.setDensity(0.012768); // g/cm^3,
  gap.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  gap.setColor(um2::yellow);
  gap.addNuclide("He4", 2.68714E-05);
  gap.populateXSec(xslib);

  // Clad
  //---------------------------------------------------------------------------
  um2::Material clad;
  clad.setName("Clad");
  clad.setDensity(6.56); // g/cm^3, (pg. 18)
  clad.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  clad.setColor(um2::slategray);
  // Number densities in atoms/b-cm from Table P4-3 (pg. 54)
  clad.addNuclide(24050, 3.30121e-06);
  clad.addNuclide(24052, 6.36606e-05);
  clad.addNuclide(24053, 7.21860e-06);
  clad.addNuclide(24054, 1.79686e-06);
  clad.addNuclide(26054, 8.68307e-06);
  clad.addNuclide(26056, 1.36306e-04);
  clad.addNuclide(26057, 3.14789e-06);
  clad.addNuclide(26058, 4.18926e-07);
  clad.addNuclide(40090, 2.18865e-02);
  clad.addNuclide(40091, 4.77292e-03);
  clad.addNuclide(40092, 7.29551e-03);
  clad.addNuclide(40094, 7.39335e-03);
  clad.addNuclide(40096, 1.19110e-03);
  clad.addNuclide(50112, 4.68066e-06);
  clad.addNuclide(50114, 3.18478e-06);
  clad.addNuclide(50115, 1.64064e-06);
  clad.addNuclide(50116, 7.01616e-05);
  clad.addNuclide(50117, 3.70592e-05);
  clad.addNuclide(50118, 1.16872e-04);
  clad.addNuclide(50119, 4.14504e-05);
  clad.addNuclide(50120, 1.57212e-04);
  clad.addNuclide(50122, 2.23417e-05);
  clad.addNuclide(50124, 2.79392e-05);
  clad.addNuclide(72174, 3.54138e-09);
  clad.addNuclide(72176, 1.16423e-07);
  clad.addNuclide(72177, 4.11686e-07);
  clad.addNuclide(72178, 6.03806e-07);
  clad.addNuclide(72179, 3.01460e-07);
  clad.addNuclide(72180, 7.76449e-07);
  clad.populateXSec(xslib);

  // Moderator
  //---------------------------------------------------------------------------
  um2::Material moderator;
  moderator.setName("Moderator");
  moderator.setDensity(0.743); // g/cm^3, Table P1-1 (pg. 20)
  moderator.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  moderator.setColor(um2::blue);
  // Number densities in atoms/b-cm from Table P4-3 (pg. 54)
  moderator.addNuclide(1001, 4.96194e-02);
  moderator.addNuclide(5010, 1.12012e-05);
  moderator.addNuclide(5011, 4.50862e-05);
  moderator.addNuclide(8016, 2.48097e-02);
  moderator.populateXSec(xslib);

  // Pyrex
  //---------------------------------------------------------------------------
  um2::Material pyrex;
  pyrex.setName("Pyrex");
  pyrex.setDensity(2.25); // g/cm^3, Table 6 (pg. 8)
  pyrex.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  pyrex.setColor(um2::orange);
  // Number densities in atoms/b-cm from Table P4-3 (pg. 55)
  pyrex.addNuclide( 5010, 9.63266e-04);
  pyrex.addNuclide( 5011, 3.90172e-03);
  pyrex.addNuclide( 8016, 4.67761e-02);
  // pyrex.addNuclide(14028, 1.81980e-02);
  // pyrex.addNuclide(14029, 9.24474e-04);
  // pyrex.addNuclide(14030, 6.10133e-04);
  pyrex.addNuclide(14000, 1.97326E-02); // Natural Silicon
  pyrex.populateXSec(xslib);

  // SS304
  //---------------------------------------------------------------------------
  um2::Material ss304;
  ss304.setName("SS304");
  ss304.setDensity(8.0); // g/cm^3, (pg. 18)
  ss304.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  ss304.setColor(um2::darkgray);
  ss304.addNuclide(6000, 3.20895e-04); // Natural Carbon
  // ss304.addNuclide(14028, 1.58197e-03);
  // ss304.addNuclide(14029, 8.03653e-05);
  // ss304.addNuclide(14030, 5.30394e-05);
  ss304.addNuclide(14000, 1.71537E-03); // Natural Silicon
  ss304.addNuclide(15031, 6.99938e-05);
  ss304.addNuclide(24050, 7.64915e-04);
  ss304.addNuclide(24052, 1.47506e-02);
  ss304.addNuclide(24053, 1.67260e-03);
  ss304.addNuclide(24054, 4.16346e-04);
  ss304.addNuclide(25055, 1.75387e-03);
  ss304.addNuclide(26054, 3.44776e-03);
  ss304.addNuclide(26056, 5.41225e-02);
  ss304.addNuclide(26057, 1.24992e-03);
  ss304.addNuclide(26058, 1.66342e-04);
  ss304.addNuclide(28058, 5.30854e-03);
  ss304.addNuclide(28060, 2.04484e-03);
  ss304.addNuclide(28061, 8.88879e-05);
  ss304.addNuclide(28062, 2.83413e-04);
  ss304.addNuclide(28064, 7.21770e-05);
  ss304.populateXSec(xslib);

  // AIC
  //---------------------------------------------------------------------------
  um2::Material aic;
  aic.setName("AIC");
  aic.setDensity(10.2); // g/cm^3, Table 8 (pg. 10)
  aic.setTemperature(565.0); // K, Table P4-1 (pg. 50)
  aic.setColor(um2::purple);
  // Number densities in atoms/b-cm from Table P4-3 (pg. 55)
  aic.addNuclide(47107, 2.36159e-02);
  aic.addNuclide(47109, 2.19403e-02);
  aic.addNuclide(48000, 2.73220e-03); // Natural Cadmium
  aic.addNuclide(49113, 3.44262e-04);
  aic.addNuclide(49115, 7.68050e-03);
  aic.populateXSec(xslib);

  //============================================================================
  // Geometry
  //============================================================================

  // Pin ID  |  Description
  // --------+----------------
  // 0       |  Fuel 2.11%
  // 1       |  Empty guide tube
  // 2       |  Empty instrument tube
  // 3       |  Fuel 2.61%
  // 4       |  Pyrex
  // 5       |  Poison (AIC)

  // Parameters for the pin-cell geometry
  Float const r_fuel = 0.4096; // Pellet radius = 0.4096 cm (pg. 4)
//  Float const r_gap = 0.418;   // Inner clad radius = 0.418 cm (pg. 4)
  Float const r_clad = 0.475;  // Outer clad radius = 0.475 cm (pg. 4)
  Float const r_gt_inner = 0.561; // Inner guide tube radius = 0.561 cm (pg. 5)
  Float const r_gt_outer = 0.602; // Outer guide tube radius = 0.602 cm (pg. 5)
  Float const r_it_inner = 0.559; // Inner instrument tube radius = 0.559 cm (pg. 5)
  Float const r_it_outer = 0.605; // Outer instrument tube radius = 0.605 cm (pg. 5)
  Float const r_pyrex_it_inner = 0.214; // Inner tube inner radius = 0.214 cm (pg. 8)
//  Float const r_pyrex_it_outer = 0.238; // Inner tube outer radius = 0.238 cm (pg. 8)
  Float const r_pyrex_inner = 0.241; // Pyrex inner radius = 0.241 cm (pg. 8)
  Float const r_pyrex_outer = 0.427; // Pyrex outer radius = 0.427 cm (pg. 8)
//  Float const r_pyrex_clad_inner = 0.437; // Clad inner radius = 0.437 cm (pg. 8)
  Float const r_pyrex_clad_outer = 0.484; // Clad outer radius = 0.484 cm (pg. 8)
  Float const r_aic = 0.382;    // Poison radius = 0.382 cm (pg. 10)
//  Float const r_aic_inner = 0.386; // Cladding inner radius = 0.386 cm (pg. 10)
  Float const r_aic_outer = 0.484; // Cladding outer radius = 0.484 cm (pg. 10)
  Float const pitch = 1.26;    // Pitch = 1.26 cm (pg. 4)
  Float const assembly_pitch = 21.50; // Assembly pitch = 21.50 cm (pg. 5)
  Float const inter_assembly_gap = 0.04; // Inter-assembly gap = 0.04 cm (pg. 5)
  um2::Vec2F const pin_size(pitch, pitch);

  // Fuel
  //---------------------------------------------------------------------------
  //um2::Vector<Float> const fuel_radii = {r_fuel, r_gap, r_clad};
  //um2::Vector<um2::Material> const fuel_2110_materials = {fuel_2110, gap, clad};
  //um2::Vector<um2::Material> const fuel_2619_materials = {fuel_2619, gap, clad};
  um2::Vector<Float> const fuel_radii = {r_fuel, r_clad};
  um2::Vector<um2::Material> const fuel_2110_materials = {fuel_2110, clad};
  um2::Vector<um2::Material> const fuel_2619_materials = {fuel_2619, clad};

  // Empty guide tube
  //---------------------------------------------------------------------------
  um2::Vector<Float> const gt_radii = {r_gt_inner, r_gt_outer};
  um2::Vector<um2::Material> const gt_materials = {moderator, clad};

  // Empty instrument tube
  //---------------------------------------------------------------------------
  um2::Vector<Float> const it_radii = {r_it_inner, r_it_outer};
  um2::Vector<um2::Material> const it_materials = {moderator, clad};

  // Pyrex
  //---------------------------------------------------------------------------
  // um2::Vector<Float> const pyrex_radii = {
  //   r_pyrex_it_inner, r_pyrex_it_outer, r_pyrex_inner, r_pyrex_outer, r_pyrex_clad_inner,
  //   r_pyrex_clad_outer, r_it_inner, r_it_outer
  // };
  // um2::Vector<um2::Material> const pyrex_materials = {
  //   gap, ss304, gap, pyrex, gap, ss304, moderator, clad
  // };
  // Extend the ss304 to touch the pyrex (eliminate the gap)
  um2::Vector<Float> const pyrex_radii = {
    r_pyrex_it_inner, r_pyrex_inner, r_pyrex_outer,
    r_pyrex_clad_outer, r_it_inner, r_it_outer
  };
  um2::Vector<um2::Material> const pyrex_materials = {
    gap, ss304, pyrex, ss304, moderator, clad
  };

  // AIC
  //---------------------------------------------------------------------------
  //um2::Vector<Float> const aic_radii = {r_aic, r_aic_inner, r_aic_outer};
  //um2::Vector<um2::Material> const aic_materials = {aic, gap, ss304};

  um2::Vector<Float> const aic_radii = {r_aic, r_aic_outer};
  um2::Vector<um2::Material> const aic_materials = {aic, ss304};

  // Materials, radii, and xy_extents for each pin
  //---------------------------------------------------------------------------
  um2::Vector<um2::Vector<um2::Material>> const materials = {
    fuel_2110_materials, gt_materials, it_materials, fuel_2619_materials, pyrex_materials,
    aic_materials
  };
  um2::Vector<um2::Vector<Float>> const radii = {
    fuel_radii, gt_radii, it_radii, fuel_radii, pyrex_radii, aic_radii
  };
  um2::Vector<um2::Vec2F> const xy_extents(6, pin_size);

  // Lattice layout (Fig. 3, pg. 5)
  um2::Vector<um2::Vector<Int>> const fuel_2110_lattice = um2::stringToLattice<Int>(R"(
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0
      0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 1 0 0 1 0 0 2 0 0 1 0 0 1 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
      0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    )");

  um2::Vector<um2::Vector<Int>> const fuel_2619_lattice = um2::stringToLattice<Int>(R"(
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 4 3 3 4 3 3 4 3 3 3 3 3
      3 3 3 4 3 3 3 3 3 3 3 3 3 4 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 4 3 3 4 3 3 4 3 3 4 3 3 4 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 4 3 3 4 3 3 2 3 3 4 3 3 4 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 4 3 3 4 3 3 4 3 3 4 3 3 4 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 4 3 3 3 3 3 3 3 3 3 4 3 3 3
      3 3 3 3 3 4 3 3 4 3 3 4 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
      3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
    )");

  um2::Vector<um2::Vector<Int>> const fuel_2110_aic_lattice = um2::stringToLattice<Int>(R"(
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 5 0 0 5 0 0 5 0 0 0 0 0
      0 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 2 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 5 0 0 5 0 0 5 0 0 5 0 0 5 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 5 0 0 0 0 0 0 0 0 0 5 0 0 0
      0 0 0 0 0 5 0 0 5 0 0 5 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    )");

  ASSERT(fuel_2110_lattice.size() == 17);
  ASSERT(fuel_2110_lattice[0].size() == 17);
  ASSERT(fuel_2619_lattice.size() == 17);
  ASSERT(fuel_2619_lattice[0].size() == 17);

  // Make the calls a bit more readable using an alias
  namespace factory = um2::gmsh::model::occ;

  // Create lattices
  factory::addCylindricalPinLattice2D(
      fuel_2110_lattice,
      xy_extents,
      radii,
      materials,
      /*offset=*/{assembly_pitch / 2 + inter_assembly_gap, inter_assembly_gap});

  factory::addCylindricalPinLattice2D(
      fuel_2619_lattice,
      xy_extents,
      radii,
      materials,
      /*offset=*/{-assembly_pitch / 2 + inter_assembly_gap, inter_assembly_gap});

  factory::addCylindricalPinLattice2D(
      fuel_2619_lattice,
      xy_extents,
      radii,
      materials,
      /*offset=*/{assembly_pitch / 2 + inter_assembly_gap, assembly_pitch + inter_assembly_gap});

  factory::addCylindricalPinLattice2D(
      fuel_2110_aic_lattice,
      xy_extents,
      radii,
      materials,
      /*offset=*/{-assembly_pitch / 2 + inter_assembly_gap, assembly_pitch + inter_assembly_gap});

  //===========================================================================
  // Overlay CMFD mesh
  //===========================================================================

  // Construct the MPACT model
  um2::mpact::Model model;
  model.addMaterial(fuel_2110);
  model.addMaterial(fuel_2619);
  model.addMaterial(gap);
  model.addMaterial(clad);
  model.addMaterial(moderator);
  model.addMaterial(pyrex);
  model.addMaterial(ss304);
  model.addMaterial(aic);

   // Add a coarse grid that evenly subdivides the domain (quarter core)
  um2::Vec2F const domain_extents(1.5 * assembly_pitch, 1.5 * assembly_pitch);
  um2::Vec2I const num_cells(num_coarse_cells, num_coarse_cells);
  model.addCoarseGrid(domain_extents, num_cells);
  um2::gmsh::model::occ::overlayCoarseGrid(model, moderator);

  //===========================================================================
  // Generate the mesh
  //===========================================================================

  um2::gmsh::model::mesh::setGlobalMeshSize(pitch / 12);
  um2::gmsh::model::mesh::generateMesh(um2::MeshType::Tri);
  um2::gmsh::write("4_2d.inp");

  //===========================================================================
  // Complete the MPACT model and write the mesh
  //===========================================================================

  model.importCoarseCellMeshes("4_2d.inp");
  model.writeCMFDInfo("4_2d_cmfd_info.xdmf");
  model.write("4_2d.xdmf", /*write_knudsen_data=*/true, /*write_xsec_data=*/true);
  um2::finalize();
  return 0;
}
