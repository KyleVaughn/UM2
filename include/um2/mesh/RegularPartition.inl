namespace um2
{
// --------------------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------------------

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RegularPartition<D, T, P>::getChild(Args... args) noexcept
    -> P &
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    assert(index[i] < this->num_cells[i]);
  }
  if constexpr (D == 2) {
    return children[index[0] + index[1] * this->num_cells[0]];
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * this->num_cells[i - 1];
    }
    Size const child_index = index.dot(exclusive_scan_prod);
    return children[child_index];
  }
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RegularPartition<D, T, P>::getChild(Args... args) const noexcept
    -> P const &
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    assert(index[i] < this->num_cells[i]);
  }
  if constexpr (D == 2) {
    return children[index[0] + index[1] * this->num_cells[0]];
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * this->num_cells[i - 1];
    }
    Size const child_index = index.dot(exclusive_scan_prod);
    return children[child_index];
  }
}

} // namespace um2
