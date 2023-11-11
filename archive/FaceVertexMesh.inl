////==============================================================================
//// getUniqueEdges
////==============================================================================
//
// template <Size N, Size D, std::floating_point T, std::signed_integral I>
// void
// getUniqueEdges(LinearPolygonMesh<N, D, T, I> const & mesh,
//               Vector<LineSegment<D, T>> & edges)
//{
//  Size const edges_per_face = N;
//  Size const nfaces = numFaces(mesh);
//  Size const non_unique_edges = nfaces * edges_per_face;
//  Vector<Vec2<I>> edge_conn_vec(non_unique_edges);
//  for (Size iface = 0; iface < nfaces; ++iface) {
//    auto const & face_conn = mesh.fv[iface];
//    for (Size iedge = 0; iedge < edges_per_face - 1; ++iedge) {
//      Vec2<I> edge_conn(face_conn[iedge], face_conn[iedge + 1]);
//      if (edge_conn[0] > edge_conn[1]) {
//        std::swap(edge_conn[0], edge_conn[1]);
//      }
//      edge_conn_vec[iface * edges_per_face + iedge] = edge_conn;
//    }
//    Vec2<I> edge_conn(face_conn[edges_per_face - 1], face_conn[0]);
//    if (edge_conn[0] > edge_conn[1]) {
//      std::swap(edge_conn[0], edge_conn[1]);
//    }
//    edge_conn_vec[iface * edges_per_face + edges_per_face - 1] = edge_conn;
//  }
//
//  // Sort the edges by the first vertex index, then the second vertex index.
//  std::sort(edge_conn_vec.begin(), edge_conn_vec.end(),
//            [](auto const & a, auto const & b) {
//              return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
//            });
//
//  // Get the unique edges.
//  auto last = std::unique(edge_conn_vec.begin(), edge_conn_vec.end());
//  Size const nunique_edges = static_cast<Size>(last - edge_conn_vec.begin());
//
//  // Copy the unique edges into the output vector.
//  edges.resize(nunique_edges);
//  for (Size i = 0; i < nunique_edges; ++i) {
//    auto const edge_conn = edge_conn_vec[i];
//    edges[i] =
//        LineSegment<D, T>(mesh.vertices[edge_conn[0]], mesh.vertices[edge_conn[1]]);
//  }
//}
//
// template <Size N, Size D, std::floating_point T, std::signed_integral I>
// void
// getUniqueEdges(QuadraticPolygonMesh<N, D, T, I> const & mesh,
//               Vector<QuadraticSegment<D, T>> & edges)
//{
//  Size const edges_per_face = N / 2;
//  Size const nfaces = numFaces(mesh);
//  Size const non_unique_edges = nfaces * edges_per_face;
//  Vector<Vec3<I>> edge_conn_vec(non_unique_edges);
//  for (Size iface = 0; iface < nfaces; ++iface) {
//    auto const & face_conn = mesh.fv[iface];
//    for (Size iedge = 0; iedge < edges_per_face - 1; ++iedge) {
//      Vec3<I> edge_conn(face_conn[iedge], face_conn[iedge + 1],
//                        face_conn[iedge + edges_per_face]);
//      if (edge_conn[0] > edge_conn[1]) {
//        std::swap(edge_conn[0], edge_conn[1]);
//      }
//      edge_conn_vec[iface * edges_per_face + iedge] = edge_conn;
//    }
//    Vec3<I> edge_conn(face_conn[edges_per_face - 1], face_conn[0], face_conn[N - 1]);
//    if (edge_conn[0] > edge_conn[1]) {
//      std::swap(edge_conn[0], edge_conn[1]);
//    }
//    edge_conn_vec[iface * edges_per_face + edges_per_face - 1] = edge_conn;
//  }
//
//  // Sort the edges by the first vertex index, then the second vertex index.
//  std::sort(edge_conn_vec.begin(), edge_conn_vec.end(),
//            [](auto const & a, auto const & b) {
//              return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]) ||
//                     (a[0] == b[0] && a[1] == b[1] && a[2] < b[2]);
//            });
//
//  // Get the unique edges.
//  auto last = std::unique(edge_conn_vec.begin(), edge_conn_vec.end());
//  Size const nunique_edges = static_cast<Size>(last - edge_conn_vec.begin());
//
//  // Copy the unique edges into the output vector.
//  edges.resize(nunique_edges);
//  for (Size i = 0; i < nunique_edges; ++i) {
//    auto const edge_conn = edge_conn_vec[i];
//    edges[i] =
//        QuadraticSegment<D, T>(mesh.vertices[edge_conn[0]], mesh.vertices[edge_conn[1]],
//                               mesh.vertices[edge_conn[2]]);
//  }
//}
//

////==============================================================================
//// getFaceAreas
////==============================================================================
//
// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// void
// FaceVertexMesh<P, N, D, T, I>::getFaceAreas(Vector<T> & areas) const noexcept
//{
//  areas.resize(numFaces());
//  for (Size i = 0; i < numFaces(); ++i) {
//    areas[i] = getFace(i).area();
//  }
//}
//
////==============================================================================
//// getUniqueEdges
////==============================================================================
//
// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
// void
// FaceVertexMesh<P, N, D, T, I>::getUniqueEdges(Vector<Edge> & edges) const noexcept
//{
//  um2::getUniqueEdges(*this, edges);
//}
//

////==============================================================================
//// printStats
////==============================================================================
//
//// template <Size P, Size N, Size D, std::floating_point T, std::signed_integral I>
//// void
//// printStats(FaceVertexMesh<P, N, D, T, I> const & mesh) noexcept
////{
////   using Edge = typename FaceVertexMesh<P, N, D, T, I>::Edge;
////   Vector<T> data;
////   std::vector<T> std_data;
////
////   // num faces & vertices
////   std::cout << "num faces: " << mesh.numFaces() << '\n';
////   std::cout << "num vertices: " << mesh.numVertices() << '\n';
////
////   // Face areas
////   mesh.getFaceAreas(data);
////   std_data.resize(static_cast<size_t>(data.size()));
////   std::copy(data.begin(), data.end(), std_data.begin());
////   std::sort(std_data.begin(), std_data.end());
////   std::cout << "\nFace areas:\n";
////   printHistogram(std_data);
////
////   // Edge lengths
////   Vector<Edge> unique_edges;
////   mesh.getUniqueEdges(unique_edges);
////   data.resize(unique_edges.size());
////   for (Size i = 0; i < unique_edges.size(); ++i) {
////     data[i] = unique_edges[i].length();
////   }
////   std_data.resize(static_cast<size_t>(data.size()));
////   std::copy(data.begin(), data.end(), std_data.begin());
////   std::sort(std_data.begin(), std_data.end());
////   std::cout << "\nEdge lengths:\n";
////   printHistogram(std_data);
////
////   // Mean chord length
////   data.resize(mesh.numFaces());
////   for (Size i = 0; i < mesh.numFaces(); ++i) {
////     data[i] = mesh.getFace(i).meanChordLength();
////   }
////   std_data.resize(static_cast<size_t>(data.size()));
////   std::copy(data.begin(), data.end(), std_data.begin());
////   std::sort(std_data.begin(), std_data.end());
////   std::cout << "\nMean chord lengths:\n";
////   printHistogram(std_data);
////
////   // Print each mcl to std cout
////   std::cout << "\nMCL:\n";
////   for (Size i = 0; i < mesh.numFaces(); ++i) {
////     std::cout << data[i] << '\n';
////   }
////   std::cout << "\nLinear MCL:\n";
////   for (Size i = 0; i < mesh.numFaces(); ++i) {
////     std::cout << linearPolygon(mesh.getFace(i)).meanChordLength() << '\n';
////   }
//// }

} // namespace um2
