#pragma once

#include <um2/config.hpp>

//#include <um2/common/bit.hpp>
//
#include <cstring> // memcpy, strcmp
//#include <string>  // std::string

namespace um2
{

// -----------------------------------------------------------------------------
// STRING
// -----------------------------------------------------------------------------
// A std::string-like class, but without an allocator template parameter.

struct String {

private:
  ////////////////////////////////
  // NOTE: ASSUMES LITTLE ENDIAN
  ///////////////////////////////
  // This should be true for all x86 processors and NVIDIA GPUs.


  // Size = 4 -> Long = 4 + 4 + 8 = 16
  // Size = 8 -> Long = 8 + 8 + 8 = 24 
  struct Long {
    Size is_long : 1;
    Size cap : sizeof(Size) * 8 - 1; 
    Size size;
    char *   data;
  };

  // 15 or 23 bytes
  static const Size min_cap = sizeof(Long) - 1; 

  struct Short
  {
    uint8_t is_long : 1;
    uint8_t size : 7;
    char data[min_cap];
  };

  struct Rep {
    union {
      Long  l;
      Short s;
    };
  };

  Rep _r;

public:

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr String() noexcept;

  template <uint64_t N>
  HOSTDEV constexpr explicit String(char const (&s)[N]) noexcept;

//  // -----------------------------------------------------------------------------
//  // Destructor
//  // -----------------------------------------------------------------------------
//
//  UM2_HOSTDEV ~String() { delete[] _data; }
//
  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLong() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Size;

//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  begin() const noexcept -> char *;


//
//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  end() const noexcept -> char *;
//
//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  cbegin() const noexcept -> char const *;
//
//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  cend() const noexcept -> char const *;
//
//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  size() const noexcept -> len_t;
//
//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  capacity() const noexcept -> len_t;
//
//  // cppcheck-suppress functionConst
//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  data() noexcept -> char *;
//
//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  data() const noexcept -> char const *;
//









//
//  UM2_HOSTDEV
//  String(String const & s);
//
//  UM2_HOSTDEV
//  String(String && s) noexcept;
//
//  explicit String(std::string const & s);
//
//  // -----------------------------------------------------------------------------
//  // Operators
//  // -----------------------------------------------------------------------------
//
//  UM2_HOSTDEV auto
//  operator=(String const & s) -> String &;
//
//  UM2_HOSTDEV auto
//  operator=(String && s) noexcept -> String &;
//
//  template <size_t N>
//  UM2_HOSTDEV auto
//  operator=(char const (&s)[N]) -> String &;
//
//  auto
//  operator=(std::string const & s) -> String &;
//
//  UM2_PURE UM2_HOSTDEV constexpr auto
//  operator==(String const & s) const noexcept -> bool;
//
//  template <size_t N>
//  UM2_PURE UM2_HOSTDEV constexpr auto
//  operator==(char const (&s)[N]) const noexcept -> bool;
//
//  UM2_PURE auto
//  operator==(std::string const & s) const noexcept -> bool;
//
//  UM2_PURE UM2_HOSTDEV constexpr auto
//  operator<(String const & s) const noexcept -> bool;
//
//  UM2_PURE UM2_HOSTDEV constexpr auto
//  operator>(String const & s) const noexcept -> bool;
//
//  UM2_PURE UM2_HOSTDEV constexpr auto
//  operator<=(String const & s) const noexcept -> bool;
//
//  UM2_PURE UM2_HOSTDEV constexpr auto
//  operator>=(String const & s) const noexcept -> bool;
//
//  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
//  operator[](len_t i) noexcept -> char &;
//
//  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
//  operator[](len_t i) const noexcept -> char const &;
//
//  // -----------------------------------------------------------------------------
//  // Methods
//  // -----------------------------------------------------------------------------
//
//  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
//  contains(char c) const noexcept -> bool;
//
//  // NOLINTNEXTLINE(readability-identifier-naming)
//  UM2_PURE [[nodiscard]] constexpr auto
//  starts_with(std::string const & s) const noexcept -> bool;
//
//  // NOLINTNEXTLINE(readability-identifier-naming)
//  UM2_PURE [[nodiscard]] constexpr auto
//  ends_with(std::string const & s) const noexcept -> bool;
//
  // -----------------------------------------------------------------------------
  // HIDDEN
  // -----------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongSize() const noexcept -> Size; 

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getShortSize() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  getLongCap() const noexcept -> Size;

  HOSTDEV [[nodiscard]] HIDDEN constexpr static auto
  getShortCap() noexcept -> Size;

}; // struct String
//
//// -----------------------------------------------------------------------------
//// Methods
//// -----------------------------------------------------------------------------
//
//UM2_PURE auto
//toString(String const & s) -> std::string;
//
} // namespace um2

#include "String.inl"
