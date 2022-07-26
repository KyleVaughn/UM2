#include "../types/types.hpp"

//template<typename T>
//static void BM_static_vector_addition(benchmark::State& state) {
//  T x1 = static_cast<T>(rand() % 10);
//  T y1 = static_cast<T>(rand() % 10);
//  T x2 = static_cast<T>(rand() % 10);
//  T y2 = static_cast<T>(rand() % 10);
//  vec2<T> static_v1(x1, y1);
//  vec2<T> static_v2(x2, y2);
//  vec2<T> static_v3;
//  benchmark::DoNotOptimize(static_v3);
//  for (auto _ : state) {
//    benchmark::DoNotOptimize(static_v3 = static_v1 + static_v2);
//    benchmark::ClobberMemory();
//  }
//}
//BENCHMARK_TEMPLATE(BM_static_vector_addition, float);
//BENCHMARK_TEMPLATE(BM_static_vector_addition, double);
//
//BENCHMARK_MAIN();
//
int main() {
  float x1 = static_cast<float>(rand() % 10);
  float y1 = static_cast<float>(rand() % 10);
  float x2 = static_cast<float>(rand() % 10);
  float y2 = static_cast<float>(rand() % 10);
  vec2<float> static_v1(x1, y1);
  vec2<float> static_v2(x2, y2);
  vec2<float> static_v3 = static_v1 + static_v2;
  int e = static_v3.x < 8 ? 1 : 0;
  return e;
}
