template <typename OtherDerived>
HOSTDEV constexpr auto
squaredDistanceTo(MatrixBase<OtherDerived> const & other) const noexcept -> Scalar
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
  return (derived() - other.derived()).squaredNorm();
}

template <typename OtherDerived>
HOSTDEV constexpr auto
distanceTo(MatrixBase<OtherDerived> const & other) const noexcept -> RealScalar
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(OtherDerived)
  return ::sqrt(derived().squaredDistanceTo(other));
}
