#include "strengthnet.h"
#include <iostream>
#include <vector>

using namespace std;

namespace {

constexpr dim3 numBlocksForTensor(const Tensor& t, dim3 blockDim) noexcept {
  dim3 numBlocks(1, 1, 1);
  numBlocks.x = (t.dims.x + blockDim.x - 1) / blockDim.x;
  numBlocks.y = (t.dims.y + blockDim.y - 1) / blockDim.y;
  return numBlocks;
}

}

namespace StrengthNetKernels
{

__global__ void scale(Tensor y, const Tensor w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= y.dims.x * y.dims.y)
    return;

  y.data[i] *= w.data[0];
}

__global__ void add(Tensor y, const Tensor a, const Tensor b) {
  assert(y.dims.x == a.dims.x);
  assert(y.dims.y == a.dims.y);
  assert(y.dims.x == b.dims.x);
  assert(y.dims.y == b.dims.y);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= y.dims.x * y.dims.y)
    return;

  y.data[i] = a.data[i] + b.data[i];
}

__global__ void dotproduct(Tensor y, const Tensor a, const Tensor b) {
  assert(a.dims.x == b.dims.x);

  float s = 0;
  for(int i = 0; i < a.dims.x; i++)
    s += a.data[i] * b.data[i];
  y.data[0] = s;
}

// y = W*x
// set blocks to partition y into squares
__global__ void matmul(Tensor y, const Tensor W, const Tensor x) {
  assert(W.dims.x - x.dims.y <= 1); // weight.dims.x must either match x dims or have exactly 1 more column of bias weights
  assert(y.dims.x == x.dims.x);     // output size must match
  assert(y.dims.y == W.dims.y);     // output size must match

  // naive implementation
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // early exit for overspilling blocks
  if(row >= y.dims.y || col >= y.dims.x)
    return;

  size_t in_stride = x.dims.y;
  size_t w_stride = W.dims.y;
  size_t out_stride = y.dims.y;

  float h = 0.0f;
  for (int i = 0; i < x.dims.y; i++) {
    h += W.data[i * w_stride + row] * x.data[col * in_stride + i];
  }
  if(W.dims.x - x.dims.y > 0) // weight matrix includes bias row
    h += W.data[(W.dims.x - 1) * w_stride + row]; // BUG? W.dims.x instead of w_stride

  y.data[col * out_stride + row] = h;
}

// y[z_index] = a*b^T
// set blocks to partition y into squares
__global__ void transposeMatmul(Tensor y, const Tensor a, const Tensor b, size_t z_index) {
  assert(a.dims.x == b.dims.x);  // input sizes must match
  assert(y.dims.x - b.dims.y <= 1);  // output size must either match or fit exactly 1 more column of bias weights
  assert(y.dims.y == a.dims.y);  // output size must match
  assert(z_index < y.dims.z); // must have room to store the result

  // naive implementation
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // early exit for overspilling blocks
  if(row >= y.dims.y || col >= y.dims.x)
    return;

  size_t b_stride = b.dims.y;
  size_t a_stride = a.dims.y;
  size_t y_stride = y.dims.y;
  size_t y_offset = z_index * y.dims.x * y.dims.y;

  float h = 0.0f;
  for (int i = 0; i < b.dims.x; i++) {
    if(col < b.dims.y)
      h += a.data[i * a_stride + row] * b.data[i * b_stride + col];
    else // construct bias row (as if b.data[...] == 1)
      h += a.data[i * a_stride + row];
  }
  y.data[y_offset + col * y_stride + row] = h;
}

__global__ void relu(Tensor h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= h.dims.x * h.dims.y)
    return;

  if(h.data[i] < 0)
    h.data[i] = 0;
}

__device__ float max(const Tensor& a) {
  extern __shared__ float buffer[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  buffer[i] = a.data[i];
  __syncthreads();

  uint n = a.dims.x;
  for (uint s = 1; s < n; s *= 2) {
      if (i + s < n && buffer[i] < buffer[i + s]) {
          buffer[i] = buffer[i + s];
      }
      __syncthreads();
  }
  return buffer[0];
}

__device__ float sum(const Tensor& a) {
  extern __shared__ float buffer[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  buffer[i] = a.data[i];
  __syncthreads();

  uint n = a.dims.x;
  for (uint s = 1; s < n; s *= 2) {
      if (i + s < n) {
          buffer[i] += buffer[i + s];
      }
      __syncthreads();
  }
  return buffer[0];
}

// uses a.dims.x shared memory floats
__global__ void softmax(Tensor a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= a.dims.x)
    return;

  a.data[i] = expf(a.data[i] - max(a));  // -max(a) improves numerical stability without changing the result
  a.data[i] /= sum(a);
}

__global__ void lossDerived(Tensor y_grad, float target, const Tensor y) {
  y_grad.data[0] = 2.f * (y.data[0] - target);
}

__global__ void softmaxDerived(Tensor z_grad, const Tensor a) {
  assert(z_grad.dims.x == a.dims.x);

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j >= z_grad.dims.x)
    return;

  float a_j = a.data[j];
  float b = 0;
  for(uint i = 0; i < a.dims.x; i++) {
    float delta = i == threadIdx.x ? 1.f : 0.f;
    b += z_grad.data[i] * a.data[i] * (delta - a_j);
  }
  __syncthreads();
  z_grad.data[j] = b;
}

// set blocks to partition x_grad into squares
__global__ void matmulDerived(Tensor x_grad, const Tensor y_grad, const Tensor W) {
  assert(W.dims.x - x_grad.dims.y <= 1);  // weight.dims.x must either match x dims or have exactly 1 more column of bias weights
  assert(y_grad.dims.x == x_grad.dims.x); // output size must match
  assert(y_grad.dims.y == W.dims.y);      // output size must match

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // early exit for overspilling blocks
  if(row >= x_grad.dims.y || col >= x_grad.dims.x)
    return;

  size_t y_stride = y_grad.dims.y;
  size_t w_stride = W.dims.y;
  size_t x_stride = x_grad.dims.y;

  float h = 0.0f;
  for (int i = 0; i < y_grad.dims.y; i++) {
    h += W.data[row * w_stride + i] * y_grad.data[col * y_stride + i];
  }
  x_grad.data[col * x_stride + row] = h;
}

// 1 block with W.dims threads
__global__ void accumulateTensorZ(Tensor W) {
  int i = threadIdx.x * W.dims.y + threadIdx.y;
  float v = 0;
  for(size_t z = 0; z < W.dims.z; z++) {
    v += W.data[z * W.dims.x * W.dims.y + i];
  }
  W.data[i] = v;
}

// 1 block with W.dims threads
__global__ void updateTensor(Tensor W, const Tensor W_grad, float weightPenalty, float learnrate) {
  assert(W.dims.x == W_grad.dims.x);
  assert(W.dims.y == W_grad.dims.y);

  int i = threadIdx.x * W.dims.y + threadIdx.y;
  float delta = 0;
  for(size_t z = 0; z < W_grad.dims.z; z++) {
    delta += W_grad.data[z * W_grad.dims.x * W_grad.dims.y + i];
  }
  delta += W.data[i] * 2 * weightPenalty;
  W.data[i] -= delta * learnrate;
}

// __global__ void forwardTanhKernel(float* softx, int ch, int row) {
//   softx[threadIdx.x * ch + row] = 10.f * tanhf(softx[threadIdx.x * ch + row]);
// }

// __global__ void backwardTanhKernel(float* ingrads, float* outputx, float* outgrads) {
//   float cosa = cosf(outputx[threadIdx.x*2 + 1]);
//   outgrads[threadIdx.x*2 + 1] = ingrads[threadIdx.x*2 + 1] * 10.f / (cosa*cosa);
// }

__global__ void sumOfSquares(Tensor y, const Tensor t) {
  extern __shared__ float buffer[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint n = t.dims.x * t.dims.y;
  if(i >= n)
    return;

  buffer[i] = t.data[i] * t.data[i];
  __syncthreads();

  for (uint s = 1; s < n; s *= 2) {
      if (i + s < n) {
          buffer[i] += buffer[i + s];
      }
      __syncthreads();
  }

  if(0 == i)
    y.data[0] = buffer[0];
}

} // end namespace StrengthNetKernels

using namespace StrengthNetKernels;

void StrengthNet::forward() {
  uint N = x.dims.x;
  dim3 blockDim1d(1024);
  dim3 blockDim2d(16, 16);

  // layer 1
  dim3 numBlocks = numBlocksForTensor(h, blockDim2d);
  matmul<<<numBlocks, blockDim2d>>>(h, W1, x);
  numBlocks = dim3((N * h.dims.y + blockDim1d.x - 1) / blockDim1d.x);
  relu<<<numBlocks, blockDim1d>>>(h);

  // layer 2
  numBlocks = numBlocksForTensor(r, blockDim2d);
  matmul<<<numBlocks, blockDim2d>>>(r, W2r, h);
  numBlocks = numBlocksForTensor(a, blockDim2d);
  matmul<<<numBlocks, blockDim2d>>>(a, W2z, h);

  // aggregate by attention
  numBlocks = numBlocksForTensor(a, blockDim1d);
  softmax<<<numBlocks, blockDim1d, N*sizeof(float)>>>(a);
  dotproduct<<<1, 1>>>(y, r, a);
}

void StrengthNet::backward(float target, size_t index) {
  assert(index < batchSize);

  uint N = x.dims.x;
  dim3 blockDim1d(1024);
  dim3 blockDim2d(16, 16);

  target = (target - 1500.f) / 500.f;
  lossDerived<<<1,1>>>(y_grad, target, y); // dL/dy

  // aggregate by attention
  r_grad.assignFrom(a); // dy/dr
  dim3 numBlocks = numBlocksForTensor(r_grad, blockDim1d);
  scale<<<numBlocks, blockDim1d>>>(r_grad, y_grad); // dL/dr = dL/dy * dy/dr

  z_grad.assignFrom(r); // dy/da
  scale<<<numBlocks, blockDim1d>>>(z_grad, y_grad); // dL/da = dL/dy * dy/da
  softmaxDerived<<<numBlocks, blockDim1d>>>(z_grad, a); // dL/dz2 = da/dz2 * dL/da

  // layer 2
  numBlocks = numBlocksForTensor(W2r_grad, blockDim2d);
  transposeMatmul<<<numBlocks, blockDim2d>>>(W2r_grad, r_grad, h, index); // dL/dW2r = dL/dr * h^T
  transposeMatmul<<<numBlocks, blockDim2d>>>(W2z_grad, z_grad, h, index); // dL/dW2z = dL/dz * h^T

  numBlocks = numBlocksForTensor(hr_grad, blockDim2d);
  matmulDerived<<<numBlocks, blockDim2d>>>(hr_grad, r_grad, W2r);
  matmulDerived<<<numBlocks, blockDim2d>>>(hz_grad, z_grad, W2z);
  numBlocks = dim3((N * h_grad.dims.y + blockDim1d.x - 1) / blockDim1d.x);
  add<<<numBlocks, blockDim1d>>>(h_grad, hr_grad, hz_grad); // dL/dh = dr/dh * dL/dr + dz/dh * dL/dz

  // layer 1
  relu<<<numBlocks, blockDim1d>>>(h_grad); // dL/dz1 = dL/dh * dh/dz1
 
  numBlocks = numBlocksForTensor(W1_grad, blockDim2d);
  transposeMatmul<<<numBlocks, blockDim2d>>>(W1_grad, h_grad, x, index); // dL/dW1 = dL/dz1 * x^T
}

void StrengthNet::mergeGrads() {
  accumulateTensorZ<<<1, {W1_grad.dims.x, W1_grad.dims.y}>>>(W1_grad);
  accumulateTensorZ<<<1, {W2r_grad.dims.x, W2r_grad.dims.y}>>>(W2r_grad);
  accumulateTensorZ<<<1, {W2z_grad.dims.x, W2z_grad.dims.y}>>>(W2z_grad);
  W1_grad.dims.z = W2r_grad.dims.z = W2z_grad.dims.z = 1;
}

void StrengthNet::update(float weightPenalty, float learnrate) {
  updateTensor<<<1, {W1.dims.x, W1.dims.y}>>>(W1, W1_grad, weightPenalty, learnrate);
  updateTensor<<<1, {W2r.dims.x, W2r.dims.y}>>>(W2r, W2r_grad, weightPenalty, learnrate);
  updateTensor<<<1, {W2z.dims.x, W2z.dims.y}>>>(W2z, W2z_grad, weightPenalty, learnrate);
}

float StrengthNet::thetaSq() const {
  float sosW1, sosW2r, sosW2z;
  Tensor result(1, 1);
  dim3 blockDim2d(16, 16);
  size_t shmemSize = (in_ch + out_ch) * hidden_ch * sizeof(float);

  dim3 numBlocks = numBlocksForTensor(W1, blockDim2d);
  sumOfSquares<<<numBlocks, blockDim2d, shmemSize>>>(result, W1);
  cudaMemcpy(&sosW1, result.data, sizeof(float), cudaMemcpyDeviceToHost);

  numBlocks = numBlocksForTensor(W2r, blockDim2d);
  sumOfSquares<<<numBlocks, blockDim2d, shmemSize>>>(result, W2r);
  cudaMemcpy(&sosW2r, result.data, sizeof(float), cudaMemcpyDeviceToHost);

  numBlocks = numBlocksForTensor(W2z, blockDim2d);
  sumOfSquares<<<numBlocks, blockDim2d, shmemSize>>>(result, W2z);
  cudaError_t status = cudaMemcpy(&sosW2z, result.data, sizeof(float), cudaMemcpyDeviceToHost);

  if(status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));

  size_t paramCount = W1.dims.x * W1.dims.y + W2r.dims.x * W2r.dims.y + W2z.dims.x * W2z.dims.y;
  return (sosW1 + sosW2r + sosW2z) / paramCount;
}

float StrengthNet::gradsSq() const {
  float sosW1, sosW2r, sosW2z;
  Tensor result(1, 1);
  dim3 blockDim2d(16, 16);
  size_t shmemSize = (in_ch + out_ch) * hidden_ch * sizeof(float);

  dim3 numBlocks = numBlocksForTensor(W1_grad, blockDim2d);
  sumOfSquares<<<numBlocks, blockDim2d, shmemSize>>>(result, W1_grad);
  cudaMemcpy(&sosW1, result.data, sizeof(float), cudaMemcpyDeviceToHost);

  numBlocks = numBlocksForTensor(W2r_grad, blockDim2d);
  sumOfSquares<<<numBlocks, blockDim2d, shmemSize>>>(result, W2r_grad);
  cudaMemcpy(&sosW2r, result.data, sizeof(float), cudaMemcpyDeviceToHost);

  numBlocks = numBlocksForTensor(W2z_grad, blockDim2d);
  sumOfSquares<<<numBlocks, blockDim2d, shmemSize>>>(result, W2z_grad);
  cudaError_t status = cudaMemcpy(&sosW2z, result.data, sizeof(float), cudaMemcpyDeviceToHost);

  if(status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));

  size_t paramCount = W1_grad.dims.x * W1_grad.dims.y + W2r_grad.dims.x * W2r_grad.dims.y + W2z_grad.dims.x * W2z_grad.dims.y;
  return (sosW1 + sosW2r + sosW2z) / paramCount;
}
