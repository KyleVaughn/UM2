#pragma once

#include <um2/common/memory.hpp>

#include <cuda/std/bit> // cuda::std::bit_ceil

#include <thrust/pair.h> // thrust::pair

//#include <cmath>            // std::abs
//#include <cstring>          // memcpy
//#include <initializer_list> // std::initializer_list

namespace um2
{

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------
// An std::vector-like class. Allocates 2^N elements, where N is the smallest 
// integer such that 2^N >= size.
//
// https://en.cppreference.com/w/cpp/container/vector

template <typename T, typename Allocator = BasicAllocator<T>>
struct Vector {

  using Ptr = T *;
  using EndCap = thrust::pair<Ptr, Allocator>;
  using AllocTraits = AllocatorTraits<Allocator>;

  private:
    Ptr _begin = nullptr;
    Ptr _end = nullptr;
    EndCap _end_cap = EndCap(nullptr, Allocator());

    // -----------------------------------------------------------------------------
    // Private methods
    // -----------------------------------------------------------------------------

    //  Allocate space for n objects
    //  throws length_error if n > max_size()
    //  throws (probably bad_alloc) if memory run out
    //  Precondition:  _begin_ == __end_ == __end_cap() == 0
    //  Precondition:  n > 0
    //  Postcondition: capacity() >= n
    //  Postcondition: size() == 0
//    constexpr void vallocate(idx_t n) {
//        if (n > max_size())
//            __throw_length_error();
//        auto __allocation = std::__allocate_at_least(__alloc(), __n);
//        __begin_ = __allocation.ptr;
//        __end_ = __allocation.ptr;
//        __end_cap() = __begin_ + __allocation.count;
//        __annotate_new(0);
//    }

////    constexpr void _clear() noexcept {__base_destruct_at_end(this->_begin);}
//
//
    //constexpr HIDDEN void __base_destruct_at_end(Ptr new_last) noexcept {
    //  Ptr soon_to_be_end = this->_end;
    //  while (new_last != soon_to_be_end) {
    //    AllocTraits::destroy(__alloc(), std::__to_address(--__soon_to_be_end));
    //  }
    //  this->_end = new_last;
    //}

    //constexpr HIDDEN void __clear() noexcept {__base_destruct_at_end(this->_begin);}

    //class _DestroyVector {
    //  private:
    //    Vector & _vec;

    //  public:
    //    constexpr HIDDEN _DestroyVector(Vector & vec) : _vec(vec) {}
    
    //    constexpr HIDDEN void operator()() {
    //      if (_vec._begin != nullptr) {
    //        _vec.__clear();
//  //          _alloc_traits::deallocate(__vec_.__alloc(), __vec_.__begin_, __vec_.capacity());
    //      }
    //    }
    //}; // class DestroyVector


  public:

    // Ignore UM2 naming convention to match std lib functions
    // NOLINTBEGIN(readability-identifier-naming)

    // -----------------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------------

    constexpr Vector() noexcept(noexcept(Allocator())) = default;

    HOSTDEV constexpr explicit Vector(Allocator const & a) noexcept;

////  HOSTDEV constexpr Vector(idx_t n, 
////                           T const & x,
////                           A const & a = A());
////
////  HOSTDEV explicit Vector(len_t n);
////
////  HOSTDEV Vector(len_t n, T const & value);
////
////    HOSTDEV constexpr Vector(Vector const & v);
////
////  HOSTDEV Vector(Vector && v) noexcept;
////
////  // cppcheck-suppress noExplicitConstructor
////  // NOLINTNEXTLINE(google-explicit-constructor)
////  Vector(std::initializer_list<T> const & list);
////
//
//    // -----------------------------------------------------------------------------
//    // Destructor
//    // -----------------------------------------------------------------------------
//    HOSTDEV constexpr ~Vector() noexcept { _DestroyVector(*this)(); }
//
  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

    PURE HOSTDEV [[nodiscard]] constexpr 
    auto get_allocator() const noexcept -> Allocator;
//
////  PURE HOSTDEV [[nodiscard]] constexpr auto size() const noexcept -> len_t;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto capacity() const noexcept -> len_t;
////
////  // cppcheck-suppress functionConst
////  PURE HOSTDEV constexpr auto data() noexcept -> T *;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto data() const noexcept -> T const *;
////
////  // cppcheck-suppress functionConst
////  PURE HOSTDEV constexpr auto begin() noexcept -> T *;
////
////  // cppcheck-suppress functionConst
////  PURE HOSTDEV constexpr auto end() noexcept -> T *;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto cbegin() const noexcept -> T const *;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto cend() const noexcept -> T const *;
////
////  // cppcheck-suppress functionConst
////  PURE HOSTDEV constexpr auto front() noexcept -> T &;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto front() const noexcept -> T const &;
////
////  NDEBUG_PURE HOSTDEV constexpr auto back() noexcept -> T &;
////
////  NDEBUG_PURE HOSTDEV [[nodiscard]] constexpr auto back() const noexcept
////      -> T const &;
////
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//  // -----------------------------------------------------------------------------
//  // Methods
//  // -----------------------------------------------------------------------------
//
////  HOSTDEV void clear() noexcept;
////
////  HOSTDEV inline void reserve(len_t n);
////
////  HOSTDEV void resize(len_t n);
////
////  HOSTDEV inline void push_back(T const & value);
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto empty() const -> bool;
////
////  HOSTDEV void insert(T const * pos, len_t n, T const & value);
////
////  HOSTDEV void insert(T const * pos, T const & value);
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto
////  contains(T const & value) const noexcept -> bool requires(!std::floating_point<T>);
////
////  // -----------------------------------------------------------------------------
////  // Operators
////  // -----------------------------------------------------------------------------
////
////  NDEBUG_PURE HOSTDEV constexpr auto operator[](len_t i) noexcept -> T &;
////
////  NDEBUG_PURE HOSTDEV constexpr auto operator[](len_t i) const noexcept
////      -> T const &;
////
////  HOSTDEV auto operator=(Vector const & v) -> Vector &;
////
////  HOSTDEV auto operator=(Vector && v) noexcept -> Vector &;
////
////  PURE HOSTDEV constexpr auto operator==(Vector const & v) const noexcept -> bool;
//
//  // NOLINTEND(readability-identifier-naming)
}; // struct Vector

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

//template <typename T>
//requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) PURE HOSTDEV
//    constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                            T epsilon = T{}) noexcept -> bool;
//
//template <typename T>
//requires(std::unsigned_integral<T>) PURE HOSTDEV
//    constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                            T epsilon = T{}) noexcept -> bool;

} // namespace um2

#include "Vector.inl"
