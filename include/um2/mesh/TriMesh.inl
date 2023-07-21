namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::numVertices() const noexcept -> Size
{
  return vertices.size();
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::numFaces() const noexcept -> Size
{
  return fv.size();
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
TriMesh<D, T, I>::face(Size i) const noexcept -> Face
{
  auto const v0 = static_cast<Size>(fv[i][0]);
  auto const v1 = static_cast<Size>(fv[i][1]);
  auto const v2 = static_cast<Size>(fv[i][2]);
  return Triangle<D, T>(vertices[v0], vertices[v1], vertices[v2]);
}

// -------------------------------------------------------------------
// Methods
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
TriMesh<D, T, I>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(vertices);
}

template <Size D, std::floating_point T, std::signed_integral I>
PURE constexpr auto
TriMesh<D, T, I>::faceContaining(Point<D, T> const & p) const noexcept -> Size
{
  static_assert(D == 2, "Only implemented for 2D meshes");
  for (Size i = 0; i < numFaces(); ++i) {
    if (face(i).contains(p)) {
      return i;
    }
  }
  assert(false);
  return -1;
}

template <Size D, std::floating_point T, std::signed_integral I>
[[nodiscard]] constexpr auto
TriMesh<D, T, I>::regularPartition(Vector<I> & face_ids_buffer,
                                   T const mesh_multiplier) const noexcept
    -> RegularPartition<D, T, I>
{
  static_assert(D == 2, "Only implemented for 2D meshes");
  auto const bb = boundingBox();

  // Determine the number of cells in each dimension.
  T const area = bb.width() * bb.height();
  T const mesh_density = mesh_multiplier * static_cast<T>(numFaces()) / area;
  T const linear_density = um2::sqrt(mesh_density);
  auto const nx = static_cast<Size>(um2::ceil(linear_density * bb.width()));
  auto const ny = static_cast<Size>(um2::ceil(linear_density * bb.height()));

  // Count the number of faces in each cell.
  T const dx = bb.width() / static_cast<T>(nx);
  T const dy = bb.height() / static_cast<T>(ny);
  RegularPartition<D, T, I> partition;
  RegularGrid<D, T> grid(bb.minima, {dx, dy}, {nx, ny});
  partition.grid = um2::move(grid);
  Size const nxny = nx * ny;
  // We allocate 1 extra element, mapping the count of face i to index i + 1, so we may
  // perform an exclusive scan to get the starting index of each cell.
  partition.children.resize(nxny + 1);
  for (Size i = 0; i < numFaces(); ++i) {
    Vec<4, Size> const cell_range = partition.getRangeContaining(face(i).boundingBox());
    // Iterate over index 1 as the fast variable
    for (Size cj = cell_range[1]; cj <= cell_range[3]; ++cj) {
      Size const row_start = cj * nx;
      for (Size ci = cell_range[0]; ci <= cell_range[2]; ++ci) {
        Size const cell_index = ci + row_start;
        ++partition.children[cell_index + 1];
      }
    }
  }
  // Exclusive scan to get the starting index of each cell.
  for (Size i = 1; i < partition.children.size(); ++i) {
    partition.children[i] += partition.children[i - 1];
  }
  // Allocate the face ids buffer.
  Size const num_cells = static_cast<Size>(partition.children.back());
  face_ids_buffer = um2::move(Vector<I>(num_cells, -1));
  // Assign the face ids to the buffer.
  for (Size i = 0; i < numFaces(); ++i) {
    Vec<4, Size> const cell_range = partition.getRangeContaining(face(i).boundingBox());
    // Iterate over index 1 as the fast variable
    for (Size cj = cell_range[1]; cj <= cell_range[3]; ++cj) {
      Size const row_start = cj * nx;
      for (Size ci = cell_range[0]; ci <= cell_range[2]; ++ci) {
        Size const cell_index = ci + row_start;
        auto next_index = static_cast<Size>(partition.children[cell_index]);
        while (face_ids_buffer[next_index] != -1) {
          ++next_index;
        }
        face_ids_buffer[next_index] = static_cast<I>(i);
      }
    }
  }
  return partition;
}

} // namespace um2
