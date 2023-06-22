#pragma once

#include <um2/common/memory.hpp>

#include <cuda/std/bit>     // cuda::std::bit_ceil
#include <cuda/std/utility> // cuda::std::pair

namespace um2
{

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------
// An std::vector-like class.
//
// https://en.cppreference.com/w/cpp/container/vector

template <typename T, typename Allocator = BasicAllocator<T>>
struct Vector {

  using Ptr = T *;
  using ConstPtr = T const *;
  using EndCap = cuda::std::pair<Ptr, Allocator>;
  using AllocTraits = AllocatorTraits<Allocator>;

private:
  Ptr _begin = nullptr;
  Ptr _end = nullptr;
  EndCap _end_cap = EndCap(nullptr, Allocator());

public:
  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr Vector() noexcept(noexcept(Allocator())) = default;

  HOSTDEV constexpr explicit Vector(Allocator const & a) noexcept;

  HOSTDEV explicit Vector(Size n);
  ////
  ////  HOSTDEV Vector(Size n, T const & value);
  ////
  HOSTDEV constexpr Vector(Vector const & v);

  HOSTDEV constexpr Vector(Vector && v) noexcept;
  ////
  ////  // cppcheck-suppress noExplicitConstructor
  ////  Vector(std::initializer_list<T> const & list);
  ////

  // -----------------------------------------------------------------------------
  // Destructor
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr ~Vector() noexcept;

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getAllocator() const noexcept -> Allocator;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  empty() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cbegin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cend() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  front() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  front() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  back() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  back() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  data() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> T const *;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  // HOSTDEV constexpr void
  // reserve(Size n);

  ////  HOSTDEV void clear() noexcept;
  ////
  ////
  ////  HOSTDEV void resize(Size n);
  ////
  ////  HOSTDEV inline void push_back(T const & value);
  ////
  ////  PURE HOSTDEV [[nodiscard]] constexpr auto empty() const -> bool;
  ////
  ////  HOSTDEV void insert(T const * pos, Size n, T const & value);
  ////
  ////  HOSTDEV void insert(T const * pos, T const & value);
  ////
  ////  PURE HOSTDEV [[nodiscard]] constexpr auto
  ////  contains(T const & value) const noexcept -> bool
  /// requires(!std::floating_point<T>);
  ////
  // -----------------------------------------------------------------------------
  // Operators
  // -----------------------------------------------------------------------------

  NDEBUG_PURE HOSTDEV constexpr auto
  // cppcheck-suppress functionConst
  operator[](Size i) noexcept -> T &;

  NDEBUG_PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> T const &;

  HOSTDEV constexpr auto
  operator=(Vector const & v) -> Vector &;

  HOSTDEV constexpr auto
  operator=(Vector && v) noexcept -> Vector &;
  ////
  ////  PURE HOSTDEV constexpr auto operator==(Vector const & v) const noexcept -> bool;
  //

private:
  // ---------------------------------------------------------------------------
  // HIDDEN
  // ---------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr HIDDEN auto
  // cppcheck-suppress functionConst
  alloc() noexcept -> Allocator &;

  PURE HOSTDEV [[nodiscard]] constexpr HIDDEN auto
  alloc() const noexcept -> Allocator const &;

  PURE HOSTDEV [[nodiscard]] constexpr HIDDEN auto
  // cppcheck-suppress functionConst
  endcap() noexcept -> Ptr &;

  PURE HOSTDEV [[nodiscard]] constexpr HIDDEN auto
  endcap() const noexcept -> Ptr const &;

  // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
  struct ConstructTransaction {

    Vector & v;
    Ptr pos;
    ConstPtr const new_end;

    HOSTDEV constexpr HIDDEN explicit ConstructTransaction(Vector & v_in, Size n)
        : v(v_in),
          pos(v_in._end),
          new_end(v_in._end + n)
    {
    }

    HOSTDEV constexpr HIDDEN ~ConstructTransaction() { v._end = pos; }

    ConstructTransaction(ConstructTransaction const &) = delete;
    auto
    operator=(ConstructTransaction const &) -> ConstructTransaction & = delete;
  };

  HOSTDEV constexpr HIDDEN void
  constructAtEnd(Size n);

  template <class InputIterator, class Sentinel>
  HOSTDEV constexpr HIDDEN void
  constructAtEnd(InputIterator first, Sentinel last, Size n);

  HOSTDEV constexpr HIDDEN void
  destructAtEnd(Ptr new_last) noexcept;

  HOSTDEV constexpr HIDDEN void
  clearMemory() noexcept;

  HOSTDEV constexpr HIDDEN void
  allocate(Size n);

  class destroy_vector
  {

    Vector & _vec;

  public:
    HOSTDEV constexpr HIDDEN explicit destroy_vector(Vector & vec_in)
        : _vec(vec_in)
    {
    }

    HOSTDEV constexpr HIDDEN void
    operator()()
    {
      if (_vec._begin != nullptr) {
        _vec.clearMemory();
        AllocTraits::deallocate(_vec.alloc(), _vec._begin, _vec.capacity());
      }
    }
  };

  template <class InputIterator, class Sentinel>
  constexpr HIDDEN void
  initWithSize(InputIterator first, Sentinel last, Size n);

}; // struct Vector

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

// template <typename T>
// requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T epsilon = T{}) noexcept -> bool;
//
// template <typename T>
// requires(std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T epsilon = T{}) noexcept -> bool;

// Vector<bool> is a specialization that is not supported
template <typename Allocator>
struct Vector<bool, Allocator> {
};

} // namespace um2

#include "Vector.inl"
