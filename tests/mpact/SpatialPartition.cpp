#include <um2/mpact/spatial_partition.hpp>
// #include <um2/mpact/io.hpp>


template <typename T, typename I>
TEST(test_make_cylindrical_pin_mesh)
    um2::mpact::SpatialPartition<T, I> model;
    std::vector<double> const radii = {0.4096, 0.475, 0.575};
    double const pitch = 1.26;
    std::vector<int> const num_rings = {3, 1, 1};
    int const na = 8;

    int id = -1;
    id = model.make_cylindrical_pin_mesh(
        radii, pitch, num_rings, na, 1);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.quad.size() == 1, "size");
    int total_rings = std::reduce(num_rings.begin(), num_rings.end());
    int nfaces = (total_rings + 1) * na; 
    ASSERT(num_faces(model.quad[0]) == nfaces, "num_faces");
    auto aabb = um2::bounding_box(model.quad[0]);
    ASSERT_APPROX(um2::width(aabb), 1.26, 1e-6, "width");
    ASSERT_APPROX(um2::height(aabb), 1.26, 1e-6, "height");
    ASSERT_APPROX(aabb.minima[0], 0, 1e-6, "minima[0]");
    ASSERT_APPROX(aabb.minima[1], 0, 1e-6, "minima[1]");

    id = model.make_cylindrical_pin_mesh(
        radii, pitch, num_rings, na, 2);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.quadratic_quad.size() == 1, "size");
    ASSERT(num_faces(model.quadratic_quad[0]) == nfaces, "num_faces");
    aabb = um2::bounding_box(model.quadratic_quad[0]);
    ASSERT_APPROX(um2::width(aabb), 1.26, 1e-6, "width");
    ASSERT_APPROX(um2::height(aabb), 1.26, 1e-6, "height");
    ASSERT_APPROX(aabb.minima[0], 0, 1e-6, "minima[0]");
    ASSERT_APPROX(aabb.minima[1], 0, 1e-6, "minima[1]");

    std::vector<um2::Material> materials;
    um2::Vector<int8_t> material_ids;
    um2::Material uo2("UO2", "forestgreen");
    um2::Material clad("Clad", "lightgray");
    um2::Material water("Water", "lightblue");
    materials.insert(materials.end(), 3 * 8, uo2);
    materials.insert(materials.end(), 1 * 8, clad);
    materials.insert(materials.end(), 2 * 8, water);
    material_ids.insert(material_ids.end(), 3 * 8, 0);
    material_ids.insert(material_ids.end(), 1 * 8, 1);
    material_ids.insert(material_ids.end(), 2 * 8, 2);
    id = model.make_coarse_cell(2, 0, materials); 
    ASSERT(id == 0, "id should be 0");
    ASSERT_APPROX(model.coarse_cells[0].dxdy[0], pitch, 1e-6, "dxdy");
    ASSERT_APPROX(model.coarse_cells[0].dxdy[1], pitch, 1e-6, "dxdy");
    ASSERT(model.coarse_cells[0].mesh_type == 2, "mesh_type");
    ASSERT(model.coarse_cells[0].mesh_id == 0, "mesh_id");
    ASSERT(model.coarse_cells[0].material_ids == material_ids, "material_ids");
END_TEST

template <typename T, typename I>
TEST(test_make_rectangular_pin_mesh)
    um2::mpact::SpatialPartition<T, I> model;
    um2::Vec2<T> dxdy(2, 1);

    int id = -1;
    int nx = 1;
    int ny = 1;
    id = model.make_rectangular_pin_mesh(dxdy, nx, ny);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.quad.size() == 1, "size");
    ASSERT(num_faces(model.quad[0]) == nx * ny, "num_faces");
    auto aabb = um2::bounding_box(model.quad[0]);
    ASSERT_APPROX(um2::width(aabb),  dxdy[0], 1e-6, "width");
    ASSERT_APPROX(um2::height(aabb), dxdy[1], 1e-6, "height");
    ASSERT_APPROX(aabb.minima[0], 0, 1e-6, "minima[0]");
    ASSERT_APPROX(aabb.minima[1], 0, 1e-6, "minima[1]");

    std::vector<um2::Material> materials(nx * ny, um2::Material("A", "red"));
    um2::Vector<int8_t> material_ids(nx * ny, 0);
    id = model.make_coarse_cell(2, 0, materials); 
    ASSERT(id == 0, "id should be 0");
    ASSERT_APPROX(model.coarse_cells[0].dxdy[0], dxdy[0], 1e-6, "dxdy");
    ASSERT_APPROX(model.coarse_cells[0].dxdy[1], dxdy[1], 1e-6, "dxdy");
    ASSERT(model.coarse_cells[0].mesh_type == 2, "mesh_type");
    ASSERT(model.coarse_cells[0].mesh_id == 0, "mesh_id");
    ASSERT(model.coarse_cells[0].material_ids == material_ids, "material_ids");
END_TEST

template <typename T, typename I>
TEST(test_make_coarse_cell)
    um2::mpact::SpatialPartition<T, I> model;
    um2::Vec2<T> const dxdy(2, 1);
    int const id = model.make_coarse_cell(dxdy);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.coarse_cells.size() == 1, "size");
    ASSERT(um2::mpact::num_unique_coarse_cells(model) == 1, "num_unique_coarse_cells");
    ASSERT(um2::is_approx(model.coarse_cells[0].dxdy, dxdy), "dxdy");
    ASSERT(model.coarse_cells[0].mesh_type == -1, "mesh_type");
    ASSERT(model.coarse_cells[0].mesh_id == -1, "mesh_id");
    ASSERT(model.coarse_cells[0].material_ids.empty(), "material_ids");
    int const id2 = model.make_coarse_cell(dxdy);
    ASSERT(id2 == 1, "id should be 1");
    ASSERT(model.coarse_cells.size() == 2, "size");
    ASSERT(um2::mpact::num_unique_coarse_cells(model) == 2, "num_unique_coarse_cells");
    ASSERT(um2::is_approx(model.coarse_cells[1].dxdy, dxdy), "dxdy");
    ASSERT(model.coarse_cells[1].mesh_type == -1, "mesh_type");
    ASSERT(model.coarse_cells[1].mesh_id == -1, "mesh_id");
    ASSERT(model.coarse_cells[1].material_ids.empty(), "material_ids");
END_TEST

template <typename T, typename I>
TEST(test_make_rtm)
    um2::mpact::SpatialPartition<T, I> model;
    um2::Vec2<T> const dxdy(2, 1);
    ASSERT(model.make_coarse_cell(dxdy) == 0, "0");
    ASSERT(model.make_coarse_cell(dxdy) == 1, "1");
    ASSERT(um2::mpact::num_unique_coarse_cells(model) == 2, "num_unique_coarse_cells");
    std::vector<std::vector<int>> const cc_ids = {{0, 1}};
    int id = model.make_rtm(cc_ids);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.rtms.size() == 1, "size");
    ASSERT(model.rtms[0].children.size() == 2, "coarse_cell_ids");
    ASSERT(model.rtms[0].children[0] == 0, "coarse_cell_ids");
    ASSERT(model.rtms[0].children[1] == 1, "coarse_cell_ids");
    ASSERT(um2::mpact::num_unique_rtms(model) == 1, "num_unique_rtms");
    um2::RectilinearGrid2<T> const & grid = model.rtms[0].grid;
    ASSERT(grid.divs[0].size() == 3, "divs[0]");
    ASSERT(grid.divs[1].size() == 2, "divs[1]");
    ASSERT_APPROX(grid.divs[0][0], 0, 1e-6, "divs[0][0]");
    ASSERT_APPROX(grid.divs[0][1], 2, 1e-6, "divs[0][1]");
    ASSERT_APPROX(grid.divs[0][2], 4, 1e-6, "divs[0][2]");
    ASSERT_APPROX(grid.divs[1][0], 0, 1e-6, "divs[1][0]");
    ASSERT_APPROX(grid.divs[1][1], 1, 1e-6, "divs[1][1]");
    model.clear();

    std::vector<std::vector<int>> const cc_ids2 = {{2, 3},
                                                   {0, 1}};
    ASSERT(model.make_coarse_cell(dxdy) == 0, "0");
    ASSERT(model.make_coarse_cell(dxdy) == 1, "1");
    ASSERT(model.make_coarse_cell(dxdy) == 2, "2");
    ASSERT(model.make_coarse_cell(dxdy) == 3, "3");
    id = model.make_rtm(cc_ids2);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.rtms.size() == 1, "size");
    ASSERT(model.rtms[0].children.size() == 4, "coarse_cell_ids");
    ASSERT(model.rtms[0].children[0] == 0, "coarse_cell_ids");
    ASSERT(model.rtms[0].children[1] == 1, "coarse_cell_ids");
    ASSERT(model.rtms[0].children[2] == 2, "coarse_cell_ids");
    ASSERT(model.rtms[0].children[3] == 3, "coarse_cell_ids");
    um2::RectilinearGrid2<T> const & grid2 = model.rtms[0].grid;
    ASSERT(grid2.divs[0].size() == 3, "divs[0]");
    ASSERT(grid2.divs[1].size() == 3, "divs[1]");
    ASSERT_APPROX(grid2.divs[0][0], 0, 1e-6, "divs[0][0]");
    ASSERT_APPROX(grid2.divs[0][1], 2, 1e-6, "divs[0][1]");
    ASSERT_APPROX(grid2.divs[0][2], 4, 1e-6, "divs[0][2]");
    ASSERT_APPROX(grid2.divs[1][0], 0, 1e-6, "divs[1][0]");
    ASSERT_APPROX(grid2.divs[1][1], 1, 1e-6, "divs[1][1]");
    ASSERT_APPROX(grid2.divs[1][2], 2, 1e-6, "divs[1][2]");
END_TEST

template <typename T, typename I>
TEST(test_make_lattice)
    um2::mpact::SpatialPartition<T, I> model;
    um2::Vec2<T> const dxdy0(3, 3);
    um2::Vec2<T> const dxdy1(4, 4);
    ASSERT(model.make_coarse_cell(dxdy0) == 0, "0");
    ASSERT(model.make_coarse_cell(dxdy1) == 1, "1");
    std::vector<std::vector<int>> const cc_ids_44 = {{0, 0, 0, 0},
                                                     {0, 0, 0, 0},
                                                     {0, 0, 0, 0},
                                                     {0, 0, 0, 0}};
    std::vector<std::vector<int>> const cc_ids_33 = {{1, 1, 1},
                                                     {1, 1, 1},
                                                     {1, 1, 1}};
    ASSERT(model.make_rtm(cc_ids_33) == 0, "0");
    ASSERT(model.make_rtm(cc_ids_44) == 1, "1");
    std::vector<std::vector<int>> const rtm_ids = {{0, 1}};
    int id = model.make_lattice(rtm_ids);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.lattices.size() == 1, "size");
    ASSERT(model.lattices[0].children.size() == 2, "rtm_ids");
    ASSERT(model.lattices[0].children[0] == 0, "rtm_ids");
    ASSERT(model.lattices[0].children[1] == 1, "rtm_ids");
    ASSERT(um2::mpact::num_unique_lattices(model) == 1, "num_unique_lattices");
    um2::RegularGrid2<T> const & grid = model.lattices[0].grid;
    ASSERT(num_xcells(grid) == 2, "num_xcells");
    ASSERT(num_ycells(grid) == 1, "num_ycells");
    ASSERT_APPROX(grid.spacing[0], 12, 1e-6, "spacing[0]");
    ASSERT_APPROX(grid.spacing[1], 12, 1e-6, "spacing[1]");
    ASSERT_APPROX(grid.minima[0], 0, 1e-6, "minima[0]");
    ASSERT_APPROX(grid.minima[1], 0, 1e-6, "minima[1]");
END_TEST

template <typename T, typename I>
TEST(test_make_assembly)
    um2::mpact::SpatialPartition<T, I> model;
    um2::Vec2<T> const dxdy(1, 1);
    ASSERT(model.make_coarse_cell(dxdy) == 0, "0");
    std::vector<std::vector<int>> const cc_ids = {{0, 0},
                                                  {0, 0}};
    ASSERT(model.make_rtm(cc_ids) == 0, "0");
    std::vector<std::vector<int>> const rtm_ids = {{0}};
    ASSERT(model.make_lattice(rtm_ids) == 0, "0");
    ASSERT(model.make_lattice(rtm_ids) == 1, "1");
    std::vector<int> const lat_ids = {0, 1, 0};
    std::vector<double> const lat_z = {0, 2, 3, 4};
    int id = model.make_assembly(lat_ids, lat_z);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.assemblies.size() == 1, "size");
    ASSERT(model.assemblies[0].children.size() == 3, "lattice_ids");
    ASSERT(model.assemblies[0].children[0] == 0, "lattice_ids");
    ASSERT(model.assemblies[0].children[1] == 1, "lattice_ids");
    ASSERT(model.assemblies[0].children[2] == 0, "lattice_ids");
    ASSERT(um2::mpact::num_unique_assemblies(model) == 1, "num_unique_assemblies");
    um2::RectilinearGrid1<T> const & grid = model.assemblies[0].grid;
    ASSERT(grid.divs[0].size() == 4, "divs");
    ASSERT_APPROX(grid.divs[0][0], 0, 1e-6, "z");
    ASSERT_APPROX(grid.divs[0][1], 2, 1e-6, "z");
    ASSERT_APPROX(grid.divs[0][2], 3, 1e-6, "z");
    ASSERT_APPROX(grid.divs[0][3], 4, 1e-6, "z");
END_TEST

template <typename T, typename I>
TEST(test_make_assembly_2d)
    um2::mpact::SpatialPartition<T, I> model;
    um2::Vec2<T> const dxdy(1, 1);
    ASSERT(model.make_coarse_cell(dxdy) == 0, "0");
    std::vector<std::vector<int>> const cc_ids = {{0, 0},
                                                  {0, 0}};
    ASSERT(model.make_rtm(cc_ids) == 0, "0");
    std::vector<std::vector<int>> const rtm_ids = {{0}};
    ASSERT(model.make_lattice(rtm_ids) == 0, "0");
    ASSERT(model.make_lattice(rtm_ids) == 1, "1");
    std::vector<int> const lat_ids = {0};
    std::vector<double> const lat_z = {-1, 1};
    int id = model.make_assembly(lat_ids, lat_z);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.assemblies.size() == 1, "size");
    ASSERT(model.assemblies[0].children.size() == 1, "lattice_ids");
    ASSERT(model.assemblies[0].children[0] == 0, "lattice_ids");
    um2::RectilinearGrid1<T> const & grid = model.assemblies[0].grid;
    ASSERT(grid.divs[0].size() == 2, "divs");
    ASSERT_APPROX(grid.divs[0][0], -1, 1e-6, "z");
    ASSERT_APPROX(grid.divs[0][1], 1, 1e-6, "z");
END_TEST

template <typename T, typename I>
TEST(test_make_core)
    um2::mpact::SpatialPartition<T, I> model;

    um2::Vec2<T> const dxdy(2, 1);
    ASSERT(model.make_coarse_cell(dxdy) == 0, "0");

    std::vector<std::vector<int>> const cc_ids = {{0}};
    ASSERT(model.make_rtm(cc_ids) == 0, "0");

    std::vector<std::vector<int>> const rtm_ids = {{0}};
    ASSERT(model.make_lattice(rtm_ids) == 0, "0");

    std::vector<int> const lat_ids1 = {0, 0, 0};
    std::vector<double> const lat_z1 = {0, 2, 3, 4};
    ASSERT(model.make_assembly(lat_ids1, lat_z1) == 0, "0");
    std::vector<int> const lat_ids2 = {0, 0};
    std::vector<double> const lat_z2 = {0, 3, 4};
    ASSERT(model.make_assembly(lat_ids2, lat_z2) == 1, "1");
    ASSERT(model.make_assembly(lat_ids1, lat_z1) == 2, "2");
    ASSERT(model.make_assembly(lat_ids2, lat_z2) == 3, "3");

    std::vector<std::vector<int>> const ass_ids = {{2, 3},
                                                   {0, 1}};
    int id = model.make_core(ass_ids);
    ASSERT(id == 0, "id should be 0");
    ASSERT(model.core.children.size() == 4, "assembly_ids");
    ASSERT(model.core.children[0] == 0, "assembly_ids");
    ASSERT(model.core.children[1] == 1, "assembly_ids");
    ASSERT(model.core.children[2] == 2, "assembly_ids");
    ASSERT(model.core.children[3] == 3, "assembly_ids");
    ASSERT(model.core.grid.divs[0].size() == 3, "divs[0]");
    ASSERT(model.core.grid.divs[1].size() == 3, "divs[1]");
    ASSERT_APPROX(model.core.grid.divs[0][0], 0, 1e-6, "divs[0][0]");
    ASSERT_APPROX(model.core.grid.divs[0][1], 2, 1e-6, "divs[0][1]");
    ASSERT_APPROX(model.core.grid.divs[0][2], 4, 1e-6, "divs[0][2]");
    ASSERT_APPROX(model.core.grid.divs[1][0], 0, 1e-6, "divs[1][0]");
    ASSERT_APPROX(model.core.grid.divs[1][1], 1, 1e-6, "divs[1][1]");
    ASSERT_APPROX(model.core.grid.divs[1][2], 2, 1e-6, "divs[1][2]");
END_TEST

template <typename T, typename I>
TEST(test_import_coarse_cells)
    typedef typename um2::mpact::SpatialPartition<T, I>::CoarseCell CoarseCell;
    um2::mpact::SpatialPartition<T, I> model;
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_rtm({{2, 2},
                    {0, 1}});
    model.make_lattice({{0}});
    model.make_assembly({0});
    model.make_core({{0}});
    model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");

    ASSERT(model.tri.size() == 2, "tri");
    CoarseCell const & cell = model.coarse_cells[0];
    ASSERT(cell.mesh_type == 1, "mesh_type");
    ASSERT(cell.mesh_id == 0, "mesh_id");
    ASSERT(cell.material_ids.size() == 2, "material_ids");
    ASSERT(cell.material_ids[0] == 1, "material_ids");
    ASSERT(cell.material_ids[1] == 2, "material_ids");
    um2::TriMesh<T, I> const & tri_mesh = model.tri[0];
    ASSERT(tri_mesh.vertices.size() == 4, "vertices");
    ASSERT(um2::is_approx(tri_mesh.vertices[0], {0, 0}), "vertices");
    ASSERT(um2::is_approx(tri_mesh.vertices[1], {1, 0}), "vertices");
    ASSERT(um2::is_approx(tri_mesh.vertices[2], {1, 1}), "vertices");
    ASSERT(um2::is_approx(tri_mesh.vertices[3], {0, 1}), "vertices");
    ASSERT(tri_mesh.fv_offsets.empty(), "fv_offsets");
    um2::Vector<I> fv_ref = {0, 1, 2, 2, 3, 0};
    ASSERT(tri_mesh.fv == fv_ref, "fv");

    CoarseCell const & cell1 = model.coarse_cells[1];
    ASSERT(cell1.mesh_type == 1, "mesh_type");
    ASSERT(cell1.mesh_id == 1, "mesh_id");
    ASSERT(cell1.material_ids.size() == 2, "material_ids");
    ASSERT(cell1.material_ids[0] == 1, "material_ids");
    ASSERT(cell1.material_ids[1] == 0, "material_ids");
    um2::TriMesh<T, I> const & tri_mesh1 = model.tri[1];
    ASSERT(tri_mesh1.vertices.size() == 4, "vertices");
    ASSERT(um2::is_approx(tri_mesh1.vertices[0], {0, 0}), "vertices");
    ASSERT(um2::is_approx(tri_mesh1.vertices[1], {0, 1}), "vertices");
    ASSERT(um2::is_approx(tri_mesh1.vertices[2], {1, 0}), "vertices");
    ASSERT(um2::is_approx(tri_mesh1.vertices[3], {1, 1}), "vertices");
    ASSERT(tri_mesh1.fv_offsets.empty(), "fv_offsets");
    fv_ref = {0, 2, 1, 2, 3, 1};
    ASSERT(tri_mesh1.fv == fv_ref, "fv");

    CoarseCell const & cell2 = model.coarse_cells[2];
    ASSERT(cell2.mesh_type == 2, "mesh_type");
    ASSERT(cell2.mesh_id == 0, "mesh_id");
    ASSERT(cell2.material_ids.size() == 1, "material_ids");
    ASSERT(cell2.material_ids[0] == 0, "material_ids");
    um2::QuadMesh<T, I> const & quad_mesh = model.quad[0];
    ASSERT(quad_mesh.vertices.size() == 4, "vertices");
    ASSERT(um2::is_approx(quad_mesh.vertices[0], {1, 0}), "vertices");
    ASSERT(um2::is_approx(quad_mesh.vertices[1], {0, 0}), "vertices");
    ASSERT(um2::is_approx(quad_mesh.vertices[2], {1, 1}), "vertices");
    ASSERT(um2::is_approx(quad_mesh.vertices[3], {0, 1}), "vertices");
    ASSERT(quad_mesh.fv_offsets.empty(), "fv_offsets");
    fv_ref = {1, 0, 2, 3};
    ASSERT(quad_mesh.fv == fv_ref, "fv");
END_TEST

template <typename T, typename I>
TEST(test_export_mesh)
    um2::mpact::SpatialPartition<T, I> model;
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_rtm({{2, 2},
                    {0, 1}});
    model.make_lattice({{0}});
    model.make_assembly({0});
    model.make_core({{0}});

    model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");
    std::string filepath = std::filesystem::current_path().string() + "/mpact_export_test/model.xdmf";
    um2::export_mesh(filepath, model); 
END_TEST

template <typename T, typename I>
TEST(test_coarse_cell_face_areas)
    um2::mpact::SpatialPartition<T, I> model;
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_rtm({{2, 2},
                    {0, 1}});
    model.make_lattice({{0}});
    model.make_assembly({0});
    model.make_core({{0}});
    model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");

    um2::Vector<T> areas;
    model.coarse_cell_face_areas(0, areas);
    ASSERT(areas.size() == 2, "areas");
    ASSERT_APPROX(areas[0], 0.5, 1e-4, "areas");
    ASSERT_APPROX(areas[1], 0.5, 1e-4, "areas");
    model.coarse_cell_face_areas(1, areas);
    ASSERT(areas.size() == 2, "areas");
    ASSERT_APPROX(areas[0], 0.5, 1e-4, "areas");
    ASSERT_APPROX(areas[1], 0.5, 1e-4, "areas");
    model.coarse_cell_face_areas(2, areas);
    ASSERT(areas.size() == 1, "areas");
    ASSERT_APPROX(areas[0], 1.0, 1e-4, "areas");
END_TEST

template <typename T, typename I>
TEST(test_coarse_cell_find_face)
    um2::mpact::SpatialPartition<T, I> model;
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_rtm({{2, 2},
                    {0, 1}});
    model.make_lattice({{0}});
    model.make_assembly({0});
    model.make_core({{0}});
    model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");

    length_t face_id = model.coarse_cell_find_face(2, 
            um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.5))); 
    ASSERT(face_id == 0, "face_id"); face_id = -2;
    face_id = model.coarse_cell_find_face(2, 
            um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(1.5)));
    ASSERT(face_id == -1, "face_id"); face_id = -2;

    face_id = model.coarse_cell_find_face(1, 
            um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.05)));
    ASSERT(face_id == 0, "face_id"); face_id = -2;
    face_id = model.coarse_cell_find_face(1, 
            um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(-0.05)));
    ASSERT(face_id == -1, "face_id"); face_id = -2;
    face_id = model.coarse_cell_find_face(1, 
            um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.95)));
    ASSERT(face_id == 1, "face_id"); face_id = -2;
END_TEST

template <typename T, typename I>
TEST(test_coarse_cell_ray_intersect)
    um2::mpact::SpatialPartition<T, I> model;
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_coarse_cell({1, 1});
    model.make_rtm({{2, 2},
                    {0, 1}});
    model.make_lattice({{0}});
    model.make_assembly({0});
    model.make_core({{0}});
    model.import_coarse_cells("./test/mpact/mesh_files/coarse_cells.inp");

    um2::Ray2<T> ray(um2::Point2<T>(static_cast<T>(0), static_cast<T>(0.5)), 
                um2::Vec2<T>(1, 0));
    int n = 8;
    T * intersections = new T[n];
    model.intersect_coarse_cell(0, ray, intersections, &n);
    ASSERT(n == 4, "intersections");
    for (int i = 0; i < n; i++)
        std::cout << intersections[i] << std::endl;
    ASSERT_APPROX(intersections[0], 0.0, 1e-4, "intersections");
    ASSERT_APPROX(intersections[1], 0.5, 1e-4, "intersections");
    ASSERT_APPROX(intersections[2], 0.5, 1e-4, "intersections");
    ASSERT_APPROX(intersections[3], 1.0, 1e-4, "intersections");

    n = 8;
    model.intersect_coarse_cell(1, ray, intersections, &n);
    ASSERT(n == 4, "intersections");
    ASSERT_APPROX(intersections[0], 0.0, 1e-4, "intersections");
    ASSERT_APPROX(intersections[1], 0.5, 1e-4, "intersections");
    ASSERT_APPROX(intersections[2], 0.5, 1e-4, "intersections");
    ASSERT_APPROX(intersections[3], 1.0, 1e-4, "intersections");

    delete [] intersections;
END_TEST

template <typename T, typename I>
TEST_SUITE(mpact_partition)
{
    TEST("make_cylindrical_pin_mesh", (test_make_cylindrical_pin_mesh<T, I>));
//    TEST("make_coarse_cell", (test_make_coarse_cell<T, I>));
//    TEST("make_rtm", (test_make_rtm<T, I>));
//    TEST("make_lattice", (test_make_lattice<T, I>));
//    TEST("make_assembly", (test_make_assembly<T, I>));
//    TEST("make_assembly_2d", (test_make_assembly_2d<T, I>));
//    TEST("make_core", (test_make_core<T, I>));
//    TEST("import_coarse_cells", (test_import_coarse_cells<T, I>));
//    TEST("export_mesh", (test_export_mesh<T, I>));
//    TEST("coarse_cell_face_areas", (test_coarse_cell_face_areas<T, I>));
//    TEST("coarse_cell_find_face", (test_coarse_cell_find_face<T, I>));
//    TEST("coarse_cell_ray_intersect", (test_coarse_cell_ray_intersect<T, I>));
}

auto main() -> int
{
//    namespace fs = std::filesystem;
//    std::string dir = fs::current_path().string() + "/mpact_export_test";
//    bool const success = fs::create_directory(dir);
//    if (!success) {
//        std::cerr << "Failed to create test directory: " << dir << std::endl;
//        return 1;
//    }

//    RUN_TEST_SUITE("mpact_partition<f,i16>", (mpact_partition<float , int16_t>));
//    RUN_TEST_SUITE("mpact_partition<f,i32>", (mpact_partition<float , int32_t>));
//    RUN_TEST_SUITE("mpact_partition<f,i64>", (mpact_partition<float , int64_t>));
//    RUN_TEST_SUITE("mpact_partition<d,i16>", (mpact_partition<double, int16_t>));
    RUN_TEST_SUITE("mpact_partition<d,i32>", (mpact_partition<double, int32_t>));
//    RUN_TEST_SUITE("mpact_partition<d,i64>", (mpact_partition<double, int64_t>));
//    fs::remove_all(dir);
    return 0;
}
