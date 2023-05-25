namespace um2
{

// We want to use capital letters for matrix variables
// NOLINTBEGIN(readability-identifier-naming)
//// -- Constructors --
//
//// From a list of columns
// template <typename T>
// template<std::same_as<Vec2<T>> ...Cols>
// requires (sizeof...(Cols) == N)
// UM2_HOSTDEV constexpr Mat2x2<T>::Mat(Cols... in_cols) : cols{in_cols...} {}

// --------------------------------------------------------------------------
// Unary operators
// --------------------------------------------------------------------------

template <typename T>
UM2_CONST UM2_HOSTDEV constexpr auto operator-(Mat2x2<T> A) -> Mat2x2<T>
{
  A.cols[0] = -A.cols[0];
  A.cols[1] = -A.cols[1];
  return A;
}

// --------------------------------------------------------------------------
// Binary operators
// --------------------------------------------------------------------------

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator+(Mat2x2<T> A, Mat2x2<T> const & B)
    -> Mat2x2<T>
{
  A.cols[0] += B.cols[0];
  A.cols[1] += B.cols[1];
  return A;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator-(Mat2x2<T> A, Mat2x2<T> const & B)
    -> Mat2x2<T>
{
  A.cols[0] -= B.cols[0];
  A.cols[1] -= B.cols[1];
  return A;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator*=(Mat2x2<T> & A, S s) -> Mat2x2<T> &
{
  A.cols[0] *= static_cast<T>(s);
  A.cols[1] *= static_cast<T>(s);
  return A;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_HOSTDEV constexpr auto
operator/=(Mat2x2<T> & A, S s) -> Mat2x2<T> &
{
  A.cols[0] /= static_cast<T>(s);
  A.cols[1] /= static_cast<T>(s);
  return A;
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator*(S s, Mat2x2<T> A) -> Mat2x2<T>
{
  return A *= static_cast<T>(s);
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator*(Mat2x2<T> A, S s) -> Mat2x2<T>
{
  return A *= static_cast<T>(s);
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator*(Mat2x2<T> const & A, Vec2<T> const & v)
    -> Vec2<T>
{
  return {A.cols[0].x * v.x + A.cols[1].x * v.y, A.cols[0].y * v.x + A.cols[1].y * v.y};
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto operator*(Mat2x2<T> const & A, Mat2x2<T> const & B)
    -> Mat2x2<T>
{
  return {A * B[0], A * B[1]};
}

template <typename T, typename S>
requires(std::same_as<T, S> || std::integral<S>) UM2_CONST UM2_HOSTDEV constexpr auto
operator/(Mat2x2<T> A, S s) -> Mat2x2<T>
{
  return A /= static_cast<T>(s);
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto det(Mat2x2<T> const & A) -> T
{
  return cross(A[0], A[1]);
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto inv(Mat2x2<T> const & A) -> Mat2x2<T>
{
  Mat2x2<T> Ainv = Mat2x2<T>{
      Vec2<T>{ A[1][1], -A[0][1]},
      Vec2<T>{-A[1][0],  A[0][0]}
  };
  Ainv /= det(A);
  return Ainv;
}
// NOLINTEND(readability-identifier-naming)
} // namespace um2
