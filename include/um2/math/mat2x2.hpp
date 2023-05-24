#pragma once

#include <um2/common/config.hpp>
#include <um2/math/vec2.hpp>
#include <um2/math/mat.hpp>

namespace um2 {

template <typename T>
using Mat2x2 = Mat<2, 2, T>;

using Mat2x2f = Mat2x2<float>;
using Mat2x2d = Mat2x2<double>;

template <typename T>
requires(std::is_arithmetic_v<T>)
struct Mat<2, 2, T> {

    // Stored column-major
    // x0 x1
    // y0 y1
    // NOLINTNEXTLINE(*-avoid-c-arrays)
    Vec2<T> cols[2];

    // -- Accessors --

    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr 
    auto operator [] (len_t i) -> Vec2<T> &
    {
        assert(0 <= i && i < 2);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return cols[i];
    }

    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr 
    auto operator [] (len_t i) const -> Vec2<T> const &
    {
        assert(0 <= i && i < 2);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        return cols[i];
    }

    // -- Constructors --

    UM2_HOSTDEV constexpr Mat() = default;
//    
//    template<std::same_as<Col> ...Cols> 
//    requires (sizeof...(Cols) == N)
//    UM2_HOSTDEV constexpr Mat(Cols... cols);

};

//// -- Aliases --
//
//template <typename T = defaultp> using Mat2x2 = Mat<2, 2, T, Q>;
//
//template <Qualifier Q = defaultp> using Mat2x2f = Mat2x2<float, Q>;
//template <Qualifier Q = defaultp> using Mat2x2d = Mat2x2<double, Q>;
//
//// -- IO --
//
//template <len_t M, len_t N, typename T>
//std::ostream & operator << (std::ostream &, Mat<M, N, T, Q> const &);
//
//// -- Unary operators --
//
//template <len_t M, len_t N, typename T>
//UM2_CONST UM2_HOSTDEV constexpr 
//Mat<M, N, T, Q> operator - (Mat<M, N, T, Q>);
//
//// -- Binary operators --
//
//template <len_t M, len_t N, typename T>
//UM2_PURE UM2_HOSTDEV constexpr 
//Mat<M, N, T, Q> operator + (Mat<M, N, T, Q>, Mat<M, N, T, Q> const &);
//
//template <len_t M, len_t N, typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Mat<M, N, T, Q> operator - (Mat<M, N, T, Q>, Mat<M, N, T, Q> const &);
//
//// -- Scalar operators --
//
//template <len_t M, len_t N, typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr 
//Mat<M, N, T, Q> & operator *= (Mat<M, N, T, Q> &, S const);
//
//template <len_t M, len_t N, typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr 
//Mat<M, N, T, Q> & operator /= (Mat<M, N, T, Q> &, S const);
//
//
//template <len_t M, len_t N, typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_CONST UM2_HOSTDEV constexpr 
//Mat<M, N, T, Q> operator * (S const, Mat<M, N, T, Q>);
//
//template <len_t M, len_t N, typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_CONST UM2_HOSTDEV constexpr 
//Mat<M, N, T, Q> operator * (Mat<M, N, T, Q>, S const);
//
//template <len_t M, len_t N, typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_CONST UM2_HOSTDEV constexpr
//Mat<M, N, T, Q> operator / (Mat<M, N, T, Q>, S const);
//
//// -- Mat2x2 --
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Vec2<T, Q> operator * (Mat2x2<T, Q> const &, Vec2<T, Q> const &);
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Mat2x2<T, Q> operator * (Mat2x2<T, Q> const &, Mat2x2<T, Q> const &);
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//T det(Mat2x2<T, Q> const &);
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Mat2x2<T, Q> inv(Mat2x2<T, Q> const &);

} // namespace um2

#include "mat2x2.inl"