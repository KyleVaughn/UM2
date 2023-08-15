namespace um2
{
// --------------------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------------------

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::xMin() const noexcept -> T
{
  return grid.xMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::yMin() const noexcept -> T
{
  return grid.yMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::zMin() const noexcept -> T
{
  return grid.zMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::dx() const noexcept -> T
{
  return grid.dx();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::dy() const noexcept -> T
{
  return grid.dy();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::dz() const noexcept -> T
{
  return grid.dz();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::numXCells() const noexcept -> Size
{
  return grid.numXCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::numYCells() const noexcept -> Size
{
  return grid.numYCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::numZCells() const noexcept -> Size
{
  return grid.numZCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::width() const noexcept -> T
{
  return grid.width();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::height() const noexcept -> T
{
  return grid.height();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::depth() const noexcept -> T
{
  return grid.depth();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::xMax() const noexcept -> T
{
  return grid.xMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::yMax() const noexcept -> T
{
  return grid.yMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::zMax() const noexcept -> T
{
  return grid.zMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::maxima() const noexcept -> Point<D, T>
{
  return grid.maxima();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RegularPartition<D, T, P>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return grid.boundingBox();
}

template <Size D, typename T, typename P>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RegularPartition<D, T, P>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  return grid.getBox(args...);
}

template <Size D, typename T, typename P>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RegularPartition<D, T, P>::getChild(Args... args) noexcept -> P &
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    assert(index[i] < grid.num_cells[i]);
  }
  if constexpr (D == 2) {
    return children[index[0] + index[1] * grid.num_cells[0]];
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * grid.num_cells[i - 1];
    }
    Size const child_index = index.dot(exclusive_scan_prod);
    return children[child_index];
  }
}

template <Size D, typename T, typename P>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    constexpr auto RegularPartition<D, T, P>::getChild(Args... args) const noexcept
    -> P const &
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    assert(index[i] < grid.num_cells[i]);
  }
  if constexpr (D == 2) {
    return children[index[0] + index[1] * grid.num_cells[0]];
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * grid.num_cells[i - 1];
    }
    Size const child_index = index.dot(exclusive_scan_prod);
    return children[child_index];
  }
}

template <Size D, typename T, typename P>
template <typename... Args>
requires(sizeof...(Args) == D) PURE HOSTDEV
    [[nodiscard]] constexpr auto RegularPartition<D, T, P>::getCellCentroid(
        Args... args) const noexcept -> Point<D, T>
{
  return grid.getCellCentroid(args...);
}

template <Size D, typename T, typename P>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularPartition<D, T, P>::getCellIndicesIntersecting(
    AxisAlignedBox<D, T> const & box) const noexcept -> Vec<2 * D, Size>
{
  return grid.getCellIndicesIntersecting(box);
}

template <Size D, typename T, typename P>
PURE HOSTDEV [[nodiscard]] constexpr auto
RegularPartition<D, T, P>::getCellIndexContaining(
    Point<D, T> const & point) const noexcept -> Vec<D, Size>
{
  return grid.getCellIndexContaining(point);
}

} // namespace um2
