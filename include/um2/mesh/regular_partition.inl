namespace um2
{

// -- Methods --

// Minima/maxima accessors.
template <len_t D, typename T, typename P>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto xMin(RegularPartition<D, T, P> const & part) -> T
{
  return xMin(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto yMin(RegularPartition<D, T, P> const & part) -> T
{
  return yMin(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto zMin(RegularPartition<D, T, P> const & part) -> T
{
  return zMin(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto xMax(RegularPartition<D, T, P> const & part) -> T
{
  return xMax(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto yMax(RegularPartition<D, T, P> const & part) -> T
{
  return yMax(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto zMax(RegularPartition<D, T, P> const & part) -> T
{
  return zMax(part.grid);
}

// Number of divisions accessors.
template <len_t D, typename T, typename P>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto numXcells(RegularPartition<D, T, P> const & part) -> len_t
{
  return numXcells(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto numYcells(RegularPartition<D, T, P> const & part) -> len_t
{
  return numYcells(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto numZcells(RegularPartition<D, T, P> const & part) -> len_t
{
  return numZcells(part.grid);
}

template <len_t D, typename T, typename P>
UM2_PURE UM2_HOSTDEV constexpr auto
numCells(RegularPartition<D, T, P> const & part) -> Vec<D, len_t>
{
  return numCells(part.grid);
}

// Width/hight/depth
template <len_t D, typename T, typename P>
requires(D >= 1) UM2_PURE UM2_HOSTDEV
    constexpr auto width(RegularPartition<D, T, P> const & part) -> T
{
  return width(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 2) UM2_PURE UM2_HOSTDEV
    constexpr auto height(RegularPartition<D, T, P> const & part) -> T
{
  return height(part.grid);
}

template <len_t D, typename T, typename P>
requires(D >= 3) UM2_PURE UM2_HOSTDEV
    constexpr auto depth(RegularPartition<D, T, P> const & part) -> T
{
  return depth(part.grid);
}

// Bounding box
template <len_t D, typename T, typename P>
UM2_PURE UM2_HOSTDEV constexpr auto
boundingBox(RegularPartition<D, T, P> const & part) -> AABox<D, T>
{
  return boundingBox(part.grid);
}

template <len_t D, typename T, typename P>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
RegularPartition<D, T, P>::getBox(len_t const i, len_t const j) const -> AABox2<T>
requires(D == 2) { return this->grid.getBox(i, j); }

template <len_t D, typename T, typename P>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
RegularPartition<D, T, P>::getChild(len_t const i, len_t const j) -> P & requires(D == 2)
{
  return this->children[j * numXcells(this->grid) + i];
}

template <len_t D, typename T, typename P>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
RegularPartition<D, T, P>::getChild(len_t const i, len_t const j) const
    -> P const & requires(D == 2)
{
  return this->children[j * num_xcells(this->grid) + i];
}

} // namespace um2