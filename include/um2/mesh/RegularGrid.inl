namespace um2
{

// --------------------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr RegularGrid<D, T>::RegularGrid(
    Point<D, T> const & minima_in, Vec<D, T> const & spacing_in,
    Vec<D, Size> const & num_cells_in) noexcept
    : minima(minima_in),
      spacing(spacing_in),
      num_cells(num_cells_in)
{
  // Ensure all spacings and num_cells are positive
  for (Size i = 0; i < D; ++i) {
    assert(spacing[i] > 0);
    assert(num_cells[i] > 0);
  }
}

template <Size D, typename T>
HOSTDEV constexpr RegularGrid<D, T>::RegularGrid(
    AxisAlignedBox<D, T> const & box) noexcept
    : minima(box.minima),
      spacing(box.maxima - box.minima)
{
  num_cells.setOnes();
}

// --------------------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::xMin() const noexcept -> T
{
  return minima[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::yMin() const noexcept -> T
{
  static_assert(2 <= D);
  return minima[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::zMin() const noexcept -> T
{
  static_assert(3 <= D);
  return minima[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dx() const noexcept -> T
{
  return spacing[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dy() const noexcept -> T
{
  static_assert(2 <= D);
  return spacing[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::dz() const noexcept -> T
{
  static_assert(3 <= D);
  return spacing[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numXCells() const noexcept -> Size
{
  return num_cells[0];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numYCells() const noexcept -> Size
{
  static_assert(2 <= D);
  return num_cells[1];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::numZCells() const noexcept -> Size
{
  static_assert(3 <= D);
  return num_cells[2];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::width() const noexcept -> T
{
  return static_cast<T>(numXCells()) * dx();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::height() const noexcept -> T
{
  static_assert(2 <= D);
  return static_cast<T>(numYCells()) * dy();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::depth() const noexcept -> T
{
  static_assert(3 <= D);
  return static_cast<T>(numZCells()) * dz();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::xMax() const noexcept -> T
{
  return xMin() + width();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::yMax() const noexcept -> T
{
  static_assert(2 <= D);
  return yMin() + height();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::zMax() const noexcept -> T
{
  static_assert(3 <= D);
  return zMin() + depth();
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::maxima() const noexcept -> Point<D, T>
{
  return minima + spacing.cwiseProduct(num_cells.template cast<T>());
} 

template <Size D, typename T>
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return AxisAlignedBox<D, T>(minima, maxima());
}

template <Size D, typename T>
template <typename ...Args>
requires (sizeof...(Args) == D)
PURE HOSTDEV constexpr auto
RegularGrid<D, T>::getBox(Args ...args) const noexcept 
     -> AxisAlignedBox<D, T>
{
   Point<D, Size> const index(args...);
   for (Size i = 0; i < D; ++i) {
     assert(index[i] < num_cells[i]);
   }
   Point<D, T> const min_point = minima + spacing.cwiseProduct(index.template cast<T>());
   Point<D, T> const max_point = min_point + spacing; 
   return AxisAlignedBox<D, T>(min_point, max_point);
 }

} // namespace um2
