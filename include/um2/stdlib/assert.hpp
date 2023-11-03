#pragma once

#include <um2/stdlib/math.hpp>

namespace um2
{

HOSTDEV constexpr void
failedAssert(char const * const file, int const line, char const * const msg)
{
  printf("Assertion failed: %s:%d: %s\n", file, line, msg);
#ifdef __CUDA_ARCH__
  __threadfence();
  asm("trap;");
#else
  exit(1);
#endif
}

HOSTDEV constexpr void
failedAssertNear(char const * const file, int const line, char const * const a,
                 char const * const b, char const * const eps)
{
  printf("Assertion failed: %s:%d: Expected %s == %s +/- %s\n", file, line, a, b, eps);
#ifdef __CUDA_ARCH__
  __threadfence();
  asm("trap;");
#else
  exit(1);
#endif
}

} // namespace um2

#if UM2_ENABLE_DBC

#define ASSERT(cond)                                                                     \
  if (!(cond)) {                                                                         \
    um2::failedAssert(__FILE__, __LINE__, #cond);                                        \
  }

#define ASSERT_NEAR(a, b, eps)                                                           \
  {                                                                                      \
    auto const a_eval = (a);                                                             \
    auto const b_eval = (b);                                                             \
    auto const diff = um2::abs(a_eval - b_eval);                                         \
    if (diff > (eps)) {                                                                  \
      um2::failedAssertNear(__FILE__, __LINE__, #a, #b, #eps);                           \
    }                                                                                    \
  }

#else

#define ASSERT(cond)
#define ASSERT_NEAR(a, b, eps)

#endif
