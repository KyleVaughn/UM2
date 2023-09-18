namespace um2
{

//==============================================================================
// clear
//==============================================================================

template <std::floating_point T>
void
Image2D<T>::clear(Color const c)
{
  um2::fill(this->children.begin(), this->children.end(), c);
}

void
writePPM(Vector<Color> const & buffer, Size nx, Size ny, std::string const & filename);

// PNG functions are not const, so we have to pass a copy of the buffer
void
writePNG(Vector<Color> buffer, Size nx, Size ny, std::string const & filename);

//==============================================================================
// write
//==============================================================================

template <std::floating_point T>
void
Image2D<T>::write(std::string const & filename) const
{
  if (filename.ends_with("png")) {
    writePNG(this->children, this->grid.num_cells[0], this->grid.num_cells[1], filename);
  } else if (filename.ends_with("ppm")) {
    writePPM(this->children, this->grid.num_cells[0], this->grid.num_cells[1], filename);
  } else {
    Log::error("Image2D::write(): unknown file extension");
    exit(EXIT_FAILURE);
  }
}

//==============================================================================
// rasterize point
//==============================================================================

// Assumes p is in the image
template <std::floating_point T>
void
Image2D<T>::rasterize(Point2<T> const & p, Color const c)
{
  auto const idx = this->getCellIndexContaining(p);
  this->getChild(idx[0], idx[1]) = c;
}

// Assumes p is in the image, but some of the disk may not be
template <std::floating_point T>
void
Image2D<T>::rasterizeAsDisk(Point2<T> const & p, T const r, Color const c)
{
  // Get the bounding box of the circle
  AxisAlignedBox2<T> const bb({p[0] - r, p[1] - r}, {p[0] + r, p[1] + r});
  auto const range = this->getCellIndicesIntersecting(bb);
  // For each cell in the bounding box, check if the centroid is in the circle.
  // If so, set the color of the cell to c. Regardless, always color the pixel
  // containing the point p.
  Size const ixmin = range[0];
  Size const iymin = range[1];
  Size const ixmax = range[2];
  Size const iymax = range[3];
  for (Size iy = iymin; iy <= iymax; ++iy) {
    // Technically, once we have ixmin and ixmax for the row, we can color all
    // pixels between ixmin and ixmax. That could be faster than this approach, especially
    // for large radii.
    for (Size ix = ixmin; ix <= ixmax; ++ix) {
      Point2<T> const centroid = this->getCellCentroid(ix, iy);
      if (p.distanceTo(centroid) <= r) {
        this->getChild(ix, iy) = c;
      }
    }
  }

  // Always color the pixel at the center of the circle
  auto const idx = this->getCellIndexContaining(p);
  this->getChild(idx[0], idx[1]) = c;
}

//==============================================================================
// rasterize line segment
//==============================================================================

// Assumes the line is in the image
template <std::floating_point T>
void
Image2D<T>::rasterize(LineSegment2<T> const & l, Color const c)
{

  // Handle vertical and horizontal lines separately

  // Want to color any pixel that the line segment intersects.
  // L(r) = P0 + r(P1 - P0), r in [0, 1]
  // We can find the set of r that intersect the x and y boundaries of each pixel.
  // Then, we can walk through the set of rx and ry in order, and color the pixel.
  // i * spacing + origin_x = x0 + rx * (x1 - x0)
  // i0 = start index (not necessarily min)
  // i0 = floor((x0 - origin_x) / spacing)
  // i1 = end index (not necessarily max)
  // i1 = floor((x1 - origin_x) / spacing)
  // Note for an image dx == dy == spacing

  T const spacing = this->dx();
  T const inv_spacing = static_cast<T>(1) / spacing;
  auto p0_shifted = l[0] - this->grid.minima;
  auto p1_shifted = l[1] - this->grid.minima;
  Vec2<T> p01 = p1_shifted - p0_shifted;
  if (p01[0] < 0) {
    um2::swap(p0_shifted, p1_shifted);
    p01 *= -1;
  }
  // Prevent division by zero
  if (um2::abs(p01[0]) < epsilonDistance<T>()) {
    p01[0] = epsilonDistance<T>();
  }
  if (um2::abs(p01[1]) < epsilonDistance<T>()) {
    p01[1] = epsilonDistance<T>();
  }
  // Get the start i and j indices
  auto i = static_cast<Size>(um2::floor(p0_shifted[0] * inv_spacing));
  auto j = static_cast<Size>(um2::floor(p0_shifted[1] * inv_spacing));
  this->getChild(i, j) = c;
  // Get the end i and j indices
  auto const iend = static_cast<Size>(um2::floor(p1_shifted[0] * inv_spacing));
  auto const jend = static_cast<Size>(um2::floor(p1_shifted[1] * inv_spacing));
  this->getChild(iend, jend) = c;
  Size const di = i < iend ? 1 : -1;
  Size const dj = j < jend ? 1 : -1;
  Vec2<T> const inv_p01(static_cast<T>(1) / p01[0], static_cast<T>(1) / p01[1]);
  T const drx = static_cast<T>(di) * spacing * inv_p01[0];
  T const dry = static_cast<T>(dj) * spacing * inv_p01[1];
  T rx = (spacing * static_cast<T>(i) - p0_shifted[0]) * inv_p01[0];
  T ry = (spacing * static_cast<T>(j) - p0_shifted[1]) * inv_p01[1];
  // Effectively set_intersection
  while (i != iend || j != jend) {
    this->getChild(i, j) = c;
    if (um2::abs(rx + drx) < um2::abs(ry + dry)) {
      i += di;
      rx += drx;
    } else {
      j += dj;
      ry += dry;
    }
  }
}
} // namespace um2
