#include "../../test_framework.hpp"

__global__ void axpy(float a, float * x, float * y) { y[threadIdx.x] = a * x[threadIdx.x]; }

TEST_CASE(testAxpy)
{
  const int len = 4;

  float a = 2.0f;
  float host_x[len] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[len];

  // Copy input data to device.
  float * device_x;
  float * device_y;
  cudaMalloc(&device_x, len * sizeof(float));
  cudaMalloc(&device_y, len * sizeof(float));
  cudaMemcpy(device_x, host_x, len * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel.
  axpy<<<1, len>>>(a, device_x, device_y);

  // Copy output data to host.
  cudaDeviceSynchronize();
  cudaMemcpy(host_y, device_y, len * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify the result.
  float host_y_expected[len] = {2.0f, 4.0f, 6.0f, 8.0f};
  for (int i = 0; i < len; ++i)
  {
    EXPECT_APPROX_EQ(host_y[i], host_y_expected[i], 1e-5f);
  }
  cudaDeviceReset();
}

auto main() -> int
{    
  bool exit_on_failure = true;
  testAxpy(exit_on_failure);                         
  return 0;
}

