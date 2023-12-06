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

// 1 block with N threads
__global__ void scale(Tensor y, const Tensor w) {
  y.data[threadIdx.x] *= w.data[0];
}

// 1 block with N threads
__global__ void add(Tensor y, const Tensor a, const Tensor b) {
  y.data[threadIdx.x] = a.data[threadIdx.x] + b.data[threadIdx.x];
}

// 1 block with N threads
__global__ void dotproduct(Tensor y, const Tensor a, const Tensor b) {
  y.data[threadIdx.x] = a.data[threadIdx.x] * b.data[threadIdx.x];
}

// y = W*x
// set blocks to partition y into squares
__global__ void matmul(Tensor y, const Tensor W, const Tensor x) {
  // naive implementation
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  assert(W.dims.x - x.dims.y <= 1); // weight.dims.x must either match x dims or have exactly 1 more column of bias weights
  assert(y.dims.x == x.dims.x);     // output size must match
  assert(y.dims.y == W.dims.y);     // output size must match

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
    h += W.data[(w_stride - 1) * w_stride + row];

  y.data[col * out_stride + row] = h;
}

// y = a*b^T
// set blocks to partition y into squares
__global__ void transposeMatmul(Tensor y, const Tensor a, const Tensor b) {
  // naive implementation
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  assert(a.dims.x == b.dims.x);  // input sizes must match
  assert(y.dims.x == b.dims.y);  // output size must match
  assert(y.dims.y == a.dims.y);  // output size must match

  // early exit for overspilling blocks
  if(row >= y.dims.y || col >= y.dims.x)
    return;

  size_t b_stride = b.dims.y;
  size_t a_stride = a.dims.y;
  size_t y_stride = y.dims.y;

  float h = 0.0f;
  for (int i = 0; i < b.dims.x; i++) {
    h += a.data[i * a_stride + row] * b.data[i * b_stride + col];
  }
  y.data[col * y_stride + row] = h;
}

// 1 block with N threads
__global__ void relu(Tensor h) {
  if(h.data[threadIdx.x] < 0)
    h.data[threadIdx.x] = 0;
}

__device__ float max(const Tensor& a) {
  extern __shared__ float buffer[];
  buffer[threadIdx.x] = a.data[threadIdx.x];
  __syncthreads();

  uint n = a.dims.x;
  for (uint s = 1; s < n; s *= 2) {
      if (threadIdx.x + s < n && buffer[threadIdx.x] < buffer[threadIdx.x + s]) {
          buffer[threadIdx.x] = buffer[threadIdx.x + s];
      }
      __syncthreads();
  }
  return buffer[0];
}

__device__ float sum(const Tensor& a) {
  extern __shared__ float buffer[];
  buffer[threadIdx.x] = a.data[threadIdx.x];
  __syncthreads();

  uint n = a.dims.x;
  for (uint s = 1; s < n; s *= 2) {
      if (threadIdx.x + s < n) {
          buffer[threadIdx.x] += buffer[threadIdx.x + s];
      }
      __syncthreads();
  }
  return buffer[0];
}

// 1 block with N threads, N shared memory floats
__global__ void softmax(Tensor a) {
  a.data[threadIdx.x] = expf(a.data[threadIdx.x] - max(a));  // -max(a) improves numerical stability without changing the result
  a.data[threadIdx.x] /= sum(a);
}

__global__ void lossDerived(Tensor y_grad, float target, const Tensor y) {
  y_grad.data[0] = 2.f * (y.data[0] - target);
}

// 1 block with N threads
__global__ void softmaxDerived(Tensor z_grad, const Tensor a) {
  assert(z_grad.dims.x == a.dims.x);

  float a_j = a.data[threadIdx.x];
  float b = 0;
  for(uint i = 0; i < a.dims.x; i++) {
    float delta = i == threadIdx.x ? 1.f : 0.f;
    b += z_grad.data[i] * a.data[i] * (delta - a_j);
  }
  __syncthreads();
  z_grad.data[threadIdx.x] = b;
}

// set blocks to partition x_grad into squares
__global__ void matmulDerived(Tensor x_grad, const Tensor y_grad, const Tensor W) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  assert(W.dims.x - x_grad.dims.y <= 1);  // weight.dims.x must either match x dims or have exactly 1 more column of bias weights
  assert(y_grad.dims.x == x_grad.dims.x); // output size must match
  assert(y_grad.dims.y == W.dims.y);      // output size must match

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
__global__ void update(Tensor W, const Tensor W_grad, float learnrate) {
  W.data[threadIdx.x * W.dims.y + threadIdx.y] -= W_grad.data[threadIdx.x * W_grad.dims.y + threadIdx.y] * learnrate;
}

// __global__ void forwardTanhKernel(float* softx, int ch, int row) {
//   softx[threadIdx.x * ch + row] = 10.f * tanhf(softx[threadIdx.x * ch + row]);
// }

// __global__ void backwardTanhKernel(float* ingrads, float* outputx, float* outgrads) {
//   float cosa = cosf(outputx[threadIdx.x*2 + 1]);
//   outgrads[threadIdx.x*2 + 1] = ingrads[threadIdx.x*2 + 1] * 10.f / (cosa*cosa);
// }

} // end namespace StrengthNetKernels

using namespace StrengthNetKernels;

void StrengthNet::forward() {
  uint N = x.dims.x;
  dim3 blockDim(16, 16);

  // layer 1
  dim3 numBlocks = numBlocksForTensor(h, blockDim);
  matmul<<<numBlocks, blockDim>>>(h, W1, x);
  relu<<<1, N * h.dims.y>>>(h);

  // layer 2
  numBlocks = numBlocksForTensor(r, blockDim);
  matmul<<<numBlocks, blockDim>>>(r, W2r, h);
  numBlocks = numBlocksForTensor(a, blockDim);
  matmul<<<numBlocks, blockDim>>>(a, W2z, h);

  // aggregate by attention
  softmax<<<1, N, N*sizeof(float)>>>(a);
  dotproduct<<<1, N>>>(y, r, a);
}

void StrengthNet::backward(float target, float learnrate) {
  uint N = x.dims.x;
  dim3 blockDim(16, 16);

  target = (target - 1500.f) / 500.f;
  lossDerived<<<1,1>>>(y_grad, target, y); // dL/dy

  // aggregate by attention
  r_grad.assignFrom(a); // dy/dr
  scale<<<1, N>>>(r_grad, y_grad); // dL/dr = dL/dy * dy/dr

  z_grad.assignFrom(r); // dy/da
  scale<<<1, N>>>(z_grad, y_grad); // dL/da = dL/dy * dy/da
  softmaxDerived<<<1, N>>>(z_grad, a); // dL/dz2 = da/dz2 * dL/da

  // layer 2
  dim3 numBlocks = numBlocksForTensor(W2r_grad, blockDim);
  transposeMatmul<<<numBlocks, blockDim>>>(W2r_grad, r_grad, h); // dL/dW2r = dL/dr * h^T
  numBlocks = numBlocksForTensor(W2z_grad, blockDim);
  transposeMatmul<<<numBlocks, blockDim>>>(W2z_grad, z_grad, h); // dL/dW2z = dL/dz * h^T

  numBlocks = numBlocksForTensor(hr_grad, blockDim);
  matmulDerived<<<numBlocks, blockDim>>>(hr_grad, r_grad, W2r);
  numBlocks = numBlocksForTensor(hz_grad, blockDim);
  matmulDerived<<<numBlocks, blockDim>>>(hz_grad, z_grad, W2z);
  add<<<1, N * h_grad.dims.y>>>(h_grad, hr_grad, hz_grad); // dL/dh = dr/dh * dL/dr + dz/dh * dL/dz

  // layer 1
  relu<<<1, N * h.dims.y>>>(h_grad); // dL/dz1 = dL/dh * dh/dz1
 
  numBlocks = numBlocksForTensor(W1_grad, blockDim);
  transposeMatmul<<<numBlocks, blockDim>>>(W1_grad, h_grad, x); // dL/dW1 = dL/dz1 * x^T

  // apply gradients
  update<<<1, {W1.dims.x, W1.dims.y}>>>(W1, W1_grad, learnrate);
  update<<<1, {W2r.dims.x, W2r.dims.y}>>>(W2r, W2r_grad, learnrate);
  update<<<1, {W2z.dims.x, W2z.dims.y}>>>(W2z, W2z_grad, learnrate);
}

