namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T, typename P>
constexpr RectilinearPartition<D, T, P>::RectilinearPartition(
    std::vector<Vec2<T>> const & dxdy, std::vector<std::vector<Size>> const & ids)
    : grid(dxdy, ids)
{
  static_assert(D == 2);
  // Flatten the ids to get the children
  // The rows are in reverse order
  Size const nx = grid.numXCells();
  Size const ny = grid.numYCells();
  children.resize(nx * ny);
  for (Size i = 0; i < ny; ++i) {
    for (Size j = 0; j < nx; ++j) {
      children[i * nx + j] =
          static_cast<P>(ids[static_cast<size_t>(ny - 1 - i)][static_cast<size_t>(j)]);
    }
  }
}

//==============================================================================
// Accessors
//==============================================================================

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::xMin() const noexcept -> T
{
  return grid.xMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::yMin() const noexcept -> T
{
  return grid.yMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::zMin() const noexcept -> T
{
  return grid.zMin();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::dx() const noexcept -> T
{
  return grid.dx();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::dy() const noexcept -> T
{
  return grid.dy();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::dz() const noexcept -> T
{
  return grid.dz();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::numXCells() const noexcept -> Size
{
  return grid.numXCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::numYCells() const noexcept -> Size
{
  return grid.numYCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::numZCells() const noexcept -> Size
{
  return grid.numZCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::numCells() const noexcept -> Vec<D, Size>
{
  return grid.numCells();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::width() const noexcept -> T
{
  return grid.width();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::height() const noexcept -> T
{
  return grid.height();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::depth() const noexcept -> T
{
  return grid.depth();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::xMax() const noexcept -> T
{
  return grid.xMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::yMax() const noexcept -> T
{
  return grid.yMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::zMax() const noexcept -> T
{
  return grid.zMax();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::maxima() const noexcept -> Point<D, T>
{
  return grid.maxima();
}

template <Size D, typename T, typename P>
PURE HOSTDEV constexpr auto
RectilinearPartition<D, T, P>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return grid.boundingBox();
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RectilinearPartition<D, T, P>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  return grid.getBox(args...);
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RectilinearPartition<D, T, P>::getFlatIndex(
    Args... args) const noexcept -> Size
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    assert(index[i] < grid.divs[i].size());
  }
  if constexpr (D == 1) {
    return index[0];
  } else if constexpr (D == 2) {
    return index[0] + index[1] * numXCells();
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * grid.num_cells[i - 1];
    }
    return index.dot(exclusive_scan_prod);
  }
}

template <Size D, typename T, typename P>
PURE HOSTDEV [[nodiscard]] constexpr auto
RectilinearPartition<D, T, P>::getFlatIndex(Vec<D, Size> const & index) const noexcept
    -> Size
{
  for (Size i = 0; i < D; ++i) {
    assert(index[i] < grid.divs[i].size());
  }
  if constexpr (D == 1) {
    return index[0];
  } else if constexpr (D == 2) {
    return index[0] + index[1] * numXCells();
  } else { // General case
    // [0, nx, nx*ny, nx*ny*nz, ...]
    Point<D, Size> exclusive_scan_prod;
    exclusive_scan_prod[0] = 1;
    for (Size i = 1; i < D; ++i) {
      exclusive_scan_prod[i] = exclusive_scan_prod[i - 1] * grid.num_cells[i - 1];
    }
    return index.dot(exclusive_scan_prod);
  }
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RectilinearPartition<D, T, P>::getChild(Args... args) noexcept
    -> P &
{
  return children[getFlatIndex(args...)];
}

template <Size D, typename T, typename P>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV
    constexpr auto RectilinearPartition<D, T, P>::getChild(Args... args) const noexcept
    -> P const &
{
  return children[getFlatIndex(args...)];
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T, typename P>
HOSTDEV constexpr void
RectilinearPartition<D, T, P>::clear() noexcept
{
  grid.clear();
  children.clear();
}

} // namespace um2
