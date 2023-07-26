namespace um2
{

void
writePPM(Vector<Color> const & buffer, Size nx, Size ny, String const & filename);

template <std::floating_point T>
template <uint64_t N>
void
Image2D<T>::write(char const (&filename)[N]) const
{
  write(String(filename));
}

template <std::floating_point T>
void
Image2D<T>::write(String const & filename) const
{
  if (filename.ends_with("ppm")) {
    writePPM(this->children, this->num_cells[0], this->num_cells[1], filename);
  }
  //  else if (filename.ends_with("png"))
  //  {
  //    write_png(filename);
  //  }
  else {
    //    spdlog::error("Image2D::write(): unknown file extension");
    exit(EXIT_FAILURE);
  }
}

template <std::floating_point T>
void
Image2D<T>::rasterize(Point2<T> const & p, T const r, Color const c)
{
  if (p[0] < this->xMin() || p[0] >= this->xMax() || p[1] < this->yMin() ||
      p[1] >= this->yMax()) {
    return;
  }

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

} // namespace um2
