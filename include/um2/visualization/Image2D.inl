namespace um2
{

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

//// Assumes some part of the line segment is in the image
//template <std::floating_point T>
//void
//Image2D<T>::rasterize(LineSegment2<T> const & line, Color const c)
//{
//  // We want to iterate over
//  T const dx = line.p1[0] - line.p0[0]; // x1 - x0
//  T const dy = line.p1[1] - line.p0[1]; // y1 - y0
//  T const d = std::sqrt(dx * dx + dy * dy); // length of line segment

} // namespace um2
