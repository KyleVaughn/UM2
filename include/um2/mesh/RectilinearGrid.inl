namespace um2
{

//==============================================================================
// Constructors
//==============================================================================

template <Size D, typename T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(AxisAlignedBox<D, T> const & box)
{
  for (Size i = 0; i < D; ++i) {
    divs[i] = {box.minima[i], box.maxima[i]};
    assert(box.minima[i] < box.maxima[i]);
  }
}

template <Size D, typename T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(
    Vector<AxisAlignedBox<D, T>> const & boxes)
{
  // Create divs by finding the unique planar divisions
  T constexpr eps = eps_distance<T>;
  for (Size i = 0; i < boxes.size(); ++i) {
    AxisAlignedBox<D, T> const & box = boxes[i];
    for (Size d = 0; d < D; ++d) {
      bool min_found = false;
      for (Size j = 0; j < divs[d].size(); ++j) {
        if (std::abs(divs[d][j] - box.minima[d]) < eps) {
          min_found = true;
          break;
        }
      }
      if (!min_found) {
        this->divs[d].push_back(box.minima[d]);
      }
      bool max_found = false;
      for (Size j = 0; j < divs[d].size(); ++j) {
        if (std::abs(divs[d][j] - box.maxima[d]) < eps) {
          max_found = true;
          break;
        }
      }
      if (!max_found) {
        this->divs[d].push_back(box.maxima[d]);
      }
    }
  }
  // We now have the unique divisions for each dimension. Sort them.
  for (Size i = 0; i < D; ++i) {
    std::sort(divs[i].begin(), divs[i].end());
  }
  // Ensure that the boxes completely cover the grid
  // all num_divs >= 2
  // n = ∏(num_divs[i] - 1)
  Size ncells_total = 1;
  for (Size i = 0; i < D; ++i) {
    assert(divs[i].size() >= 2);
    ncells_total *= divs[i].size() - 1;
  }
  assert(ncells_total == boxes.size());
}

template <Size D, typename T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(
    std::vector<Vec2<T>> const & dxdy, std::vector<std::vector<Size>> const & ids)
{
  static_assert(D == 2);
  // Convert the dxdy to AxisAlignedBoxes
  size_t const nrows = ids.size();
  size_t const ncols = ids[0].size();
  // Ensure that each row has the same number of columns
  for (size_t i = 1; i < nrows; ++i) {
    assert(ids[i].size() == ncols);
    for (size_t j = 0; j < ncols; ++j) {
      assert(ids[i][j] >= 0);
    }
  }
  Vector<AxisAlignedBox<D, T>> boxes(static_cast<Size>(nrows * ncols));
  T y = 0;
  // Iterate rows in reverse order
  for (size_t i = 0; i < nrows; ++i) {
    std::vector<Size> const & row = ids[nrows - i - 1];
    Vec2<T> lo(static_cast<T>(0), y);
    for (size_t j = 0; j < ncols; ++j) {
      Size const id = row[j];
      Vec2<T> const & dxdy_ij = dxdy[static_cast<size_t>(id)];
      Vec2<T> const hi = lo + dxdy_ij;
      boxes[static_cast<Size>(i * ncols + j)].minima = lo;
      boxes[static_cast<Size>(i * ncols + j)].maxima = hi;
      lo[0] = hi[0];
    }
    y += dxdy[static_cast<size_t>(row[0])][1];
  }
  new (this) RectilinearGrid(boxes);
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::xMin() const noexcept -> T
{
  return divs[0].front();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::yMin() const noexcept -> T
{
  static_assert(2 <= D);
  return divs[1].front();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::zMin() const noexcept -> T
{
  static_assert(3 <= D);
  return divs[2].front();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::minima() const noexcept -> Vec<D, T>
{
  Vec<D, T> mins;
  for (Size i = 0; i < D; ++i) {
    mins[i] = divs[i].front();
  }
  return mins;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::xMax() const noexcept -> T
{
  return divs[0].back();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::yMax() const noexcept -> T
{
  static_assert(2 <= D);
  return divs[1].back();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::zMax() const noexcept -> T
{
  static_assert(3 <= D);
  return divs[2].back();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::maxima() const noexcept -> Vec<D, T>
{
  Vec<D, T> maxs;
  for (Size i = 0; i < D; ++i) {
    maxs[i] = divs[i].back();
  }
  return maxs;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numXCells() const noexcept -> Size
{
  static_assert(1 <= D);
  return divs[0].size() - 1;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numYCells() const noexcept -> Size
{
  static_assert(2 <= D);
  return divs[1].size() - 1;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numZCells() const noexcept -> Size
{
  static_assert(3 <= D);
  return divs[2].size() - 1;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::numCells() const noexcept -> Vec<D, Size>
{
  Vec<D, Size> num_cells;
  for (Size i = 0; i < D; ++i) {
    num_cells[i] = divs[i].size() - 1;
  }
  return num_cells;
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::width() const noexcept -> T
{
  return xMax() - xMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::height() const noexcept -> T
{
  static_assert(2 <= D);
  return yMax() - yMin();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::depth() const noexcept -> T
{
  static_assert(3 <= D);
  return zMax() - zMin();
}

//==============================================================================
// Methods
//==============================================================================

template <Size D, typename T>
HOSTDEV constexpr void
RectilinearGrid<D, T>::clear() noexcept
{
  for (Size i = 0; i < D; ++i) {
    divs[i].clear();
  }
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RectilinearGrid<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return {minima(), maxima()};
}

template <Size D, typename T>
template <typename... Args>
  requires(sizeof...(Args) == D)
PURE HOSTDEV constexpr auto RectilinearGrid<D, T>::getBox(Args... args) const noexcept
    -> AxisAlignedBox<D, T>
{
  Point<D, Size> const index{args...};
  for (Size i = 0; i < D; ++i) {
    assert(index[i] + 1 < divs[i].size());
  }
  AxisAlignedBox<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result.minima[i] = divs[i][index[i]];
    result.maxima[i] = divs[i][index[i] + 1];
  }
  return result;
}

} // namespace um2
