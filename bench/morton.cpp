#include <benchmark/benchmark.h>
#include <um2/geometry/morton_sort_points.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define DIM 2
#define UINT_TYPE uint32_t
#define FLOAT_TYPE float 
#define NPOINTS 1 << 20

//um2::Point<DIM, FLOAT_TYPE> random_point() {
//    static thrust::default_random_engine rng;
//    static thrust::uniform_real_distribution<FLOAT_TYPE> dist(0, 4);
//    um2::Point<DIM, FLOAT_TYPE> p;
//    for (int i = 0; i < DIM; ++i) {
//        p[i] = dist(rng);
//    }
//    return p;
//}

namespace bm = benchmark;

static void basic_morton_encode2(benchmark::State& state) {
    thrust::device_vector<UINT_TYPE> inputs(state.range(0));
    thrust::device_vector<UINT_TYPE> outputs(state.range(0));
    thrust::sequence(inputs.begin(), inputs.end());
    thrust::fill(outputs.begin(), outputs.end(), 0);
    struct EncodeTwo {
        __host__ __device__
        UINT_TYPE operator()(UINT_TYPE x, UINT_TYPE y) {
            return um2::morton_encode(x, y);
        }
    } encode_two;
    for (auto _ : state) {
        thrust::transform(inputs.begin(), inputs.end(), 
                          inputs.begin(), 
                          outputs.begin(), 
                          encode_two);
    }
}

static void basic_morton_encode2_raw(benchmark::State& state) {
    UINT_TYPE * inputs = new UINT_TYPE[state.range(0)];
    UINT_TYPE * outputs = new UINT_TYPE[state.range(0)];
    thrust::sequence(inputs, inputs + state.range(0));
    thrust::fill(outputs, outputs + state.range(0), 0);
    struct EncodeTwo {
        __host__ __device__
        UINT_TYPE operator()(UINT_TYPE x, UINT_TYPE y) {
            return um2::morton_encode(x, y);
        }
    } encode_two;
    for (auto _ : state) {
        thrust::transform(inputs, inputs + state.range(0),
                          inputs,
                          outputs,
                          encode_two);
    }
}

//static void morton_encode_point(benchmark::State& state) {
//    thrust::host_vector<um2::Point<DIM, FLOAT_TYPE>> host_inputs(state.range(0));
//    thrust::device_vector<um2::Point<DIM, FLOAT_TYPE>> inputs(state.range(0));
//    thrust::device_vector<UINT_TYPE> outputs(state.range(0));
//    thrust::generate(host_inputs.begin(), host_inputs.end(), random_point);
//    thrust::copy(host_inputs.begin(), host_inputs.end(), inputs.begin());
//    thrust::fill(outputs.begin(), outputs.end(), 0);
//    um2::Vec<DIM, FLOAT_TYPE> invs;
//    for (int i = 0; i < DIM; ++i) {
//        invs[i] = static_cast<FLOAT_TYPE>(1) / 4; 
//    }
//    struct EncodePoint {
//
//        um2::Vec<DIM, FLOAT_TYPE> const invs;
//
//        __device__
//        UINT_TYPE operator()(um2::Point<DIM, FLOAT_TYPE> p) {
//            return um2::morton_encode<UINT_TYPE>(p, invs);
//        }
//    } encode_point{invs};
//    for (auto _ : state) {
//        thrust::transform(inputs.begin(), inputs.end(), 
//                          outputs.begin(), 
//                          encode_point);
//    }
//}

BENCHMARK(basic_morton_encode2)->RangeMultiplier(2)->Range(1 << 8, NPOINTS);
BENCHMARK(basic_morton_encode2_raw)->RangeMultiplier(2)->Range(1 << 8, NPOINTS);
//BENCHMARK(morton_encode_point)->RangeMultiplier(2)->Range(1 << 8, NPOINTS);

BENCHMARK_MAIN();
