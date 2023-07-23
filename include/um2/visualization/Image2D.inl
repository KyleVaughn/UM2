namespace um2
{

void writePPM(Vector<Color> const & buffer, Size nx, Size ny, String const & filename);

template <typename T>
template <uint64_t N>
void
Image2D<T>::write(char const (&filename)[N]) const
{
  write(String(filename));
}

template <typename T>
void
Image2D<T>::write(String const & filename) const
{
  if (filename.ends_with("ppm"))
  {
    writePPM(this->children, this->num_cells[0], this->num_cells[1], filename);
  }
//  else if (filename.ends_with("png"))
//  {
//    write_png(filename);
//  }
  else
  {
    fprintf(stderr, "Image2D::write(): unknown file extension\n");
    exit(EXIT_FAILURE);
  } 
}

} // namespace um2
