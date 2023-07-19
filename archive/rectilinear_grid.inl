namespace um2
{

template <len_t D, typename T>
UM2_HOSTDEV void
RectilinearGrid<D, T>::clear()
{
  for (len_t i = 0; i < D; ++i) {
    this->divs[i].clear();
  }
}

// -- Constructors --

template <len_t D, typename T>
UM2_HOSTDEV constexpr RectilinearGrid<D, T>::RectilinearGrid(AABox<D, T> const & box)
{
  for (len_t i = 0; i < D; ++i) {
    this->divs[i] = {box.minima[i], box.maxima[i]};
    assert(box.minima[i] < box.maxima[i]);
  }
}

template <len_t D, typename T>
UM2_HOSTDEV constexpr RectilinearGrid<D, T>::RectilinearGrid(
    AABox<D, T> const * const boxes, len_t const n)
{
  // Create divs by finding the unique planar divisions
  T const eps = static_cast<T>(1e-4);
  for (len_t i = 0; i < n; ++i) {
    AABox<D, T> const & box = boxes[i];
    for (len_t d = 0; d < D; ++d) {
      bool min_found = false;
      for (len_t j = 0; j < this->divs[d].size(); ++j) {
        if (std::abs(this->divs[d][j] - box.minima[d]) < eps) {
          min_found = true;
          break;
        }
      }
      if (!min_found) {
        this->divs[d].push_back(box.minima[d]);
      }
      bool max_found = false;
      for (len_t j = 0; j < this->divs[d].size(); ++j) {
        if (std::abs(this->divs[d][j] - box.maxima[d]) < eps) {
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
  for (len_t i = 0; i < D; ++i) {
    thrust::sort(thrust::seq, this->divs[i].begin(), this->divs[i].end());
  }
  // Ensure that the boxes completely cover the grid
  // all num_divs >= 2
  // n = ∏(num_divs[i] - 1)
  len_t ncells_total = 1;
  for (len_t i = 0; i < D; ++i) {
    assert(this->divs[i].size() >= 2);
    ncells_total *= this->divs[i].size() - 1;
  }
  assert(ncells_total == n);
}

template <len_t D, typename T>
UM2_HOSTDEV constexpr RectilinearGrid<D, T>::RectilinearGrid(
    Vector<AABox<D, T>> const & boxes)
{
  new (this) RectilinearGrid(boxes.data(), boxes.size());
}

template <len_t D, typename T>
constexpr RectilinearGrid<D, T>::RectilinearGrid(
    std::vector<Vec2<T>> const & dxdy,
    std::vector<std::vector<int>> const & ids) requires(D == 2)
{
  // Convert the dxdy to AABoxes
  size_t const nrows = ids.size();
  size_t const ncols = ids[0].size();
  // Ensure that each row has the same number of columns
  for (size_t i = 1; i < nrows; ++i) {
    assert(ids[i].size() == ncols);
    for (size_t j = 0; j < ncols; ++j) {
      assert(ids[i][j] >= 0);
    }
  }
  Vector<AABox<D, T>> boxes(static_cast<len_t>(nrows * ncols));
  T y = 0;
  // Iterate rows in reverse order
  for (size_t i = 0; i < nrows; ++i) {
    std::vector<int> const & row = ids[nrows - i - 1];
    Vec2<T> minima(static_cast<T>(0), y);
    for (size_t j = 0; j < ncols; ++j) {
      int const id = row[j];
      Vec2<T> const & dxdy_ij = dxdy[static_cast<size_t>(id)];
      Vec2<T> const maxima = minima + dxdy_ij;
      boxes[static_cast<len_t>(i * ncols + j)].minima = minima;
      boxes[static_cast<len_t>(i * ncols + j)].maxima = maxima;
      minima.data()[0] = maxima[0];
    }
    y += dxdy[static_cast<size_t>(row[0])][1];
  }
  new (this) RectilinearGrid(boxes);
}

// Minima/maxima accessors.
template <len_t D, typename T>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto xMin(RectilinearGrid<D, T> const & grid) -> T
{
  return grid.divs[0].front();
}

template <len_t D, typename T>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto yMin(RectilinearGrid<D, T> const & grid) -> T
{
  return grid.divs[1].front();
}

template <len_t D, typename T>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto zMin(RectilinearGrid<D, T> const & grid) -> T
{
  return grid.divs[2].front();
}

template <len_t D, typename T>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto xMax(RectilinearGrid<D, T> const & grid) -> T
{
  return grid.divs[0].back();
}

template <len_t D, typename T>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto yMax(RectilinearGrid<D, T> const & grid) -> T
{
  return grid.divs[1].back();
}

template <len_t D, typename T>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto zMax(RectilinearGrid<D, T> const & grid) -> T
{
  return grid.divs[2].back();
}

// Number of divisions accessors.
template <len_t D, typename T>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto numXcells(RectilinearGrid<D, T> const & grid) -> len_t
{
  return grid.divs[0].size() - 1;
}

template <len_t D, typename T>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto numYcells(RectilinearGrid<D, T> const & grid) -> len_t
{
  return grid.divs[1].size() - 1;
}

template <len_t D, typename T>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto numZcells(RectilinearGrid<D, T> const & grid) -> len_t
{
  return grid.divs[2].size() - 1;
}

template <len_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr auto
numCells(RectilinearGrid<D, T> const & grid) -> Vec<D, len_t>
{
  Vec<D, len_t> num_cells;
  for (len_t i = 0; i < D; ++i) {
    num_cells.data()[i] = grid.divs[i].size() - 1;
  }
  return num_cells;
}

// Width/hight/depth
template <len_t D, typename T>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto width(RectilinearGrid<D, T> const & grid) -> T
{
  return xMax(grid) - xMin(grid);
}

template <len_t D, typename T>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto height(RectilinearGrid<D, T> const & grid) -> T
{
  return yMax(grid) - yMin(grid);
}

template <len_t D, typename T>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto depth(RectilinearGrid<D, T> const & grid) -> T
{
  return zMax(grid) - zMin(grid);
}

// Bounding box
template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
boundingBox(RectilinearGrid<D, T> const & grid) -> AABox<D, T>
{
  assert(1 <= D && D <= 3);
  if constexpr (D == 1) {
    return AABox<D, T>(Point<D, T>(xMin(grid)), Point<D, T>(xMax(grid)));
  } else if constexpr (D == 2) {
    return AABox<D, T>(Point<D, T>(xMin(grid), yMin(grid)),
                       Point<D, T>(xMax(grid), yMax(grid)));
  } else {
    return AABox<D, T>(Point<D, T>(xMin(grid), yMin(grid), zMin(grid)),
                       Point<D, T>(xMax(grid), yMax(grid), zMax(grid)));
  }
}

template <len_t D, typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
RectilinearGrid<D, T>::getBox(len_t const i, len_t const j) const -> AABox2<T>
requires(D == 2)
{
  assert(i + 1 < this->divs[0].size());
  assert(j + 1 < this->divs[1].size());
  return {
      {    this->divs[0][i],     this->divs[1][j]},
      {this->divs[0][i + 1], this->divs[1][j + 1]}
  };
}

} // namespace um2