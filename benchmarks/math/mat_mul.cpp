//=============================================================================
// Results
//=============================================================================
// CPU: i7-12800H
// Seems like using the fixed size matrix is faster up until N = 32.
// clang-format off
// matMul<2, double>          0.426 ns        0.426 ns   1000000000 bytes_per_second=139.91Gi/s items_per_second=2.3473G/s
// matMul<3, double>           5.35 ns         5.35 ns    128858505 bytes_per_second=37.6046Gi/s items_per_second=186.933M/s
// matMul<4, double>           1.85 ns         1.85 ns    378889896 bytes_per_second=257.065Gi/s items_per_second=539.105M/s
// matMul<6, double>           13.0 ns         13.0 ns     53742149 bytes_per_second=123.63Gi/s items_per_second=76.8211M/s
// matMul<8, double>           15.1 ns         15.1 ns     46968492 bytes_per_second=253.308Gi/s items_per_second=66.4031M/s
// matMul<16, double>           139 ns          139 ns      4980334 bytes_per_second=219.24Gi/s items_per_second=7.18406M/s
// matMul<32, double>          2919 ns         2918 ns       241901 bytes_per_second=83.6533Gi/s items_per_second=342.644k/s
// matMul<64, double>         22650 ns        22649 ns        30914 bytes_per_second=86.2346Gi/s items_per_second=44.1521k/s
// matrixMul<2, double>        63.2 ns         63.2 ns     11029743 bytes_per_second=965.477Mi/s items_per_second=15.8184M/s
// matrixMul<3, double>        70.1 ns         70.1 ns      9971407 bytes_per_second=2.86856Gi/s items_per_second=14.2597M/s
// matrixMul<4, double>        71.2 ns         71.2 ns      9782414 bytes_per_second=6.69614Gi/s items_per_second=14.0428M/s
// matrixMul<6, double>        86.9 ns         86.9 ns      8059774 bytes_per_second=18.5232Gi/s items_per_second=11.5099M/s
// matrixMul<8, double>        95.1 ns         95.1 ns      7308714 bytes_per_second=40.1045Gi/s items_per_second=10.5132M/s
// matrixMul<16, double>        242 ns          242 ns      2902943 bytes_per_second=126.302Gi/s items_per_second=4.13867M/s
// matrixMul<32, double>       1255 ns         1255 ns       555935 bytes_per_second=194.505Gi/s items_per_second=796.692k/s
// matrixMul<64, double>       9444 ns         9443 ns        74245 bytes_per_second=206.838Gi/s items_per_second=105.901k/s
// clang-format on

#include "../helpers.hpp"

#include <um2/config.hpp>
#include <um2/math/mat.hpp>
#include <um2/math/matrix.hpp>

#include <benchmark/benchmark.h>

template <Int N, class T>
void
matMul(benchmark::State & state)
{
  um2::Mat<N, N, T> a;
  um2::Mat<N, N, T> b;
  um2::Mat<N, N, T> c;
  for (Int i = 0; i < N; ++i) {
    for (Int j = 0; j < N; ++j) {
      a(i, j) = randomFloat<T>();
      b(i, j) = randomFloat<T>();
    }
  }
  for (auto s : state) {
    c = a * b;
    benchmark::DoNotOptimize(c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * N * N * N * static_cast<Int>(sizeof(T)));
}

template <Int N, class T>
void
matrixMul(benchmark::State & state)
{
  um2::Matrix<T> a(N, N);
  um2::Matrix<T> b(N, N);
  um2::Matrix<T> c(N, N);
  for (Int i = 0; i < N; ++i) {
    for (Int j = 0; j < N; ++j) {
      a(i, j) = randomFloat<T>();
      b(i, j) = randomFloat<T>();
    }
  }
  for (auto s : state) {
    matmul(c, a, b);
    benchmark::DoNotOptimize(c);
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * N * N * N * static_cast<Int>(sizeof(T)));
}

BENCHMARK_TEMPLATE(matMul, 2, double);
BENCHMARK_TEMPLATE(matMul, 3, double);
BENCHMARK_TEMPLATE(matMul, 4, double);
BENCHMARK_TEMPLATE(matMul, 6, double);
BENCHMARK_TEMPLATE(matMul, 8, double);
BENCHMARK_TEMPLATE(matMul, 16, double);
BENCHMARK_TEMPLATE(matMul, 32, double);
BENCHMARK_TEMPLATE(matMul, 64, double);

BENCHMARK_TEMPLATE(matrixMul, 2, double);
BENCHMARK_TEMPLATE(matrixMul, 3, double);
BENCHMARK_TEMPLATE(matrixMul, 4, double);
BENCHMARK_TEMPLATE(matrixMul, 6, double);
BENCHMARK_TEMPLATE(matrixMul, 8, double);
BENCHMARK_TEMPLATE(matrixMul, 16, double);
BENCHMARK_TEMPLATE(matrixMul, 32, double);
BENCHMARK_TEMPLATE(matrixMul, 64, double);

BENCHMARK_MAIN();
