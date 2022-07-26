#include "../types/types.hpp"

template<typename T>
static void BM_static_vector_addition(benchmark::State& state) {
  T x1 = static_cast<T>(rand() % 10);
  T y1 = static_cast<T>(rand() % 10);
  T x2 = static_cast<T>(rand() % 10);
  T y2 = static_cast<T>(rand() % 10);
  vec2<T> static_v1(x1, y1);
  vec2<T> static_v2(x2, y2);
  volatile vec2<T> static_v3;
  benchmark::DoNotOptimize(static_v3);
  for (auto _ : state) {
    benchmark::DoNotOptimize(static_v3 = static_v1 + static_v2);
    benchmark::ClobberMemory();
  }
}
BENCHMARK_TEMPLATE(BM_static_vector_addition, float);
BENCHMARK_TEMPLATE(BM_static_vector_addition, double);

BENCHMARK_MAIN();