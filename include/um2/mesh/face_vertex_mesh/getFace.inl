namespace um2
{

// We specialize the method for each mesh to avoid default initialization
// of the polygon. This is important for performance.

// -------------------------------------------------------------------
// TriMesh
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(TriMesh<D, T, I> const & mesh, Size i) noexcept -> Triangle<D, T>
{
  return Triangle<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])], 
                        mesh.vertices[static_cast<Size>(mesh.fv[i][1])], 
                        mesh.vertices[static_cast<Size>(mesh.fv[i][2])]);
}

// -------------------------------------------------------------------
// QuadMesh
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadMesh<D, T, I> const & mesh, Size i) noexcept -> Quadrilateral<D, T>
{
  return Quadrilateral<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])], 
                             mesh.vertices[static_cast<Size>(mesh.fv[i][1])], 
                             mesh.vertices[static_cast<Size>(mesh.fv[i][2])], 
                             mesh.vertices[static_cast<Size>(mesh.fv[i][3])]);
}

// -------------------------------------------------------------------
// QuadraticTriMesh
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadraticTriMesh<D, T, I> const & mesh, Size i) noexcept -> QuadraticTriangle<D, T>
{
  return QuadraticTriangle<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])], 
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][1])], 
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][2])], 
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][3])], 
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][4])], 
                                 mesh.vertices[static_cast<Size>(mesh.fv[i][5])]);
}

// -------------------------------------------------------------------
// QuadraticQuadMesh
// -------------------------------------------------------------------

template <Size D, std::floating_point T, std::signed_integral I>
PURE HOSTDEV constexpr auto
getFace(QuadraticQuadMesh<D, T, I> const & mesh, Size i) noexcept -> QuadraticQuadrilateral<D, T>
{
  return QuadraticQuadrilateral<D, T>(mesh.vertices[static_cast<Size>(mesh.fv[i][0])], 
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][1])], 
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][2])], 
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][3])], 
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][4])], 
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][5])], 
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][6])], 
                                      mesh.vertices[static_cast<Size>(mesh.fv[i][7])]); 
}

} // namespace um2
