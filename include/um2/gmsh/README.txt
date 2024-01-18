We want to use um2::String and um2::Vector to be consistent with the rest of
UM2 and promote compatability with GPU code. However, Gmsh uses std::string and
std::vector. Therefore, developers should implement functions that take
std::string and std::vector as arguments, but also provide functions that take
um2::String and um2::Vector as arguments.

The recommended way to do this is to use the following pattern:

foo(std::string const & s) {
  // do something with s
}

foo(um2::String const & s) {
  // Use implicit conversion to std::string
  return foo(s.c_str());
}

Note: Gmsh uses
  - std::string
  - std::vector
  - int
  - double
