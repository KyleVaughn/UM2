#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/algorithm.hpp> // copy
#include <um2/stdlib/math.hpp>      // min
#include <um2/stdlib/memory.hpp>    // addressof
#include <um2/stdlib/utility.hpp>   // move

#include <bit>     // std::endian::native, std::endian::big
#include <cstring> // memcpy, strcmp
#include <string>  // std::string

namespace um2
{

//==============================================================================
// STRING
//==============================================================================
// A std::string-like class, but without an allocator template parameter.

static_assert(std::endian::native == std::endian::little,
              "Only little endian is supported.");

struct String {

private:
  ////////////////////////////////
  // NOTE: ASSUMES LITTLE ENDIAN
  ///////////////////////////////
  // This should be true for all x86 processors and NVIDIA GPUs.

  // Heap-allocated string representation.
  // 24 bytes
  struct Long {
    uint64_t is_long : 1;
    uint64_t cap : 63;
    uint64_t size;
    char * data;
  };

  static uint64_t constexpr min_cap = sizeof(Long) - 1;

  // Stack-allocated string representation.
  struct Short {
    uint8_t is_long : 1;
    uint8_t size : 7;
    char data[min_cap];
  };

  // Raw representation of the string.
  struct Raw {
    uint64_t raw[3];
  };

  // Union of all representations.
  struct Rep {
    union {
      Long l;
      Short s;
      Raw r;
    };
  };

  Rep _r;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  HOSTDEV constexpr String() noexcept;

  HOSTDEV constexpr String(String const & s) noexcept;

  HOSTDEV constexpr String(String && s) noexcept;

  // NOLINTBEGIN(google-explicit-constructor); justification: match std::string
  template <uint64_t N>
  HOSTDEV constexpr String(char const (&s)[N]) noexcept;

  HOSTDEV constexpr String(char const * s) noexcept;
  // NOLINTEND(google-explicit-constructor)

  template <std::integral T>
  explicit constexpr String(T x) noexcept;

  template <std::floating_point T>
  explicit constexpr String(T x) noexcept;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV constexpr ~String() noexcept
  {
    if (isLong()) {
      ::operator delete(_r.l.data);
    }
  }

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLong() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> char *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> char const *;

  //==============================================================================
  // Operators
  //==============================================================================

  HOSTDEV constexpr auto
  operator=(String const & s) noexcept -> String &;

  HOSTDEV constexpr auto
  operator=(String && s) noexcept -> String &;

  constexpr auto
  operator=(std::string const & s) noexcept -> String &;

  constexpr auto
  operator=(std::string && s) noexcept -> String &;

  template <uint64_t N>
  HOSTDEV constexpr auto
  operator=(char const (&s)[N]) noexcept -> String &;

  PURE HOSTDEV constexpr auto
  operator==(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator!=(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator<(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator<=(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator>(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator>=(String const & s) const noexcept -> bool;

  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> char &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> char const &;

  HOSTDEV constexpr auto
  operator+=(String const & s) noexcept -> String &;

  //==============================================================================
  // Methods
  //==============================================================================
  // NOLINTBEGIN(readability-identifier-naming); justification: match std::string

  PURE HOSTDEV [[nodiscard]] constexpr auto
  compare(String const & s) const noexcept -> int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  c_str() const noexcept -> char const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  starts_with(String const & s) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  ends_with(String const & s) const noexcept -> bool;

  template <uint64_t N>
  PURE HOSTDEV [[nodiscard]] auto
  ends_with(char const (&s)[N]) const noexcept -> bool;

  // NOLINTEND(readability-identifier-naming)

  //==============================================================================
  // HIDDEN
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongSize() const noexcept -> uint64_t;

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getShortSize() const noexcept -> uint8_t;

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongCap() const noexcept -> uint64_t;

  HOSTDEV [[nodiscard]] HIDDEN constexpr static auto
  getShortCap() noexcept -> uint64_t;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongPointer() noexcept -> char *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongPointer() const noexcept -> char const *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getShortPointer() noexcept -> char *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getShortPointer() const noexcept -> char const *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getPointer() noexcept -> char *;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getPointer() const noexcept -> char const *;

  CONST HOSTDEV HIDDEN static constexpr auto
  fitsInShort(uint64_t n) noexcept -> bool;

  HOSTDEV HIDDEN constexpr void
  initShort(uint64_t n) noexcept;

  HOSTDEV HIDDEN constexpr void
  initLong(uint64_t n) noexcept;

}; // struct String

template <typename T>
constexpr auto
toString(T const & t) noexcept -> String;

HOSTDEV constexpr auto
operator+(String l, String const & r) noexcept -> String;

template <uint64_t N>
HOSTDEV constexpr auto
operator+(String l, char const (&r)[N]) noexcept -> String;

template <uint64_t N>
HOSTDEV constexpr auto
operator+(char const (&l)[N], String r) noexcept -> String;

} // namespace um2

#include "String.inl"
