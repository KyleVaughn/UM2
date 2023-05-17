#include <um2/common/string.hpp>

namespace um2
{

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

UM2_HOSTDEV String::String(String const & s)
{
  this->_size = s._size;
  this->_capacity = s._capacity;
  this->_data = new char8_t[static_cast<size_t>(this->_capacity)];
  memcpy(this->_data, s._data, static_cast<size_t>(this->_size + 1));
}

String::String(std::string const & s)
{
  this->_size = static_cast<len_t>(s.size());
  this->_capacity = static_cast<len_t>(bit_ceil(s.size() + 1));
  this->_data = new char8_t[static_cast<size_t>(this->_capacity)];
  memcpy(this->_data, s.data(), static_cast<size_t>(this->_size + 1));
}

// --------------------------------------------------------------------------
// Operators
// --------------------------------------------------------------------------

UM2_HOSTDEV String & String::operator=(String const & s)
{
  if (this != &s) {
    len_t const sizep1 = s._size + 1;
    if (this->_capacity < sizep1) {
      delete[] this->_data;
      this->_capacity = static_cast<len_t>(bit_ceil(sizep1));
      this->_data = new char8_t[bit_ceil(sizep1)];
    }
    this->_size = s._size;
    memcpy(this->_data, s._data, static_cast<size_t>(sizep1));
  }
  return *this;
}

String & String::operator=(std::string const & s)
{
  size_t const sizep1 = s.size() + 1;
  if (this->_capacity < static_cast<len_t>(sizep1)) {
    delete[] this->_data;
    this->_capacity = static_cast<len_t>(bit_ceil(sizep1));
    this->_data = new char8_t[bit_ceil(sizep1)];
  }
  this->_size = static_cast<len_t>(s.size());
  memcpy(this->_data, s.data(), s.size() + 1);
  return *this;
}

UM2_PURE UM2_HOSTDEV bool String::operator<(String const & s) const
{
  return this->compare(s) < 0;
}

UM2_PURE UM2_HOSTDEV bool String::operator>(String const & s) const
{
  return this->compare(s) > 0;
}

UM2_PURE UM2_HOSTDEV bool String::operator<=(String const & s) const
{
  return this->compare(s) <= 0;
}

UM2_PURE UM2_HOSTDEV bool String::operator>=(String const & s) const
{
  return this->compare(s) >= 0;
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

#ifdef __CUDA_ARCH__
UM2_PURE UM2_DEVICE int String::compare(String const & v) const
{
  char8_t const * s1 = this->_data;
  char8_t const * s2 = v._data;
  while (*s1 && (*s1 == *s2)) {
    ++s1;
    ++s2;
  }
  return *reinterpret_cast<unsigned char const *>(s1) -
         *reinterpret_cast<unsigned char const *>(s2);
}
#else
UM2_PURE UM2_HOST int String::compare(String const & v) const
{
  // Reinterpret char8_t as char
  return strcmp(reinterpret_cast<char const *>(this->_data),
                reinterpret_cast<char const *>(v._data));
}
#endif

} // namespace um2
