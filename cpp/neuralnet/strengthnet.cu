#include "strengthnet.h"
#include <iostream>
#include <vector>
#include <cmath>

namespace {

constexpr dim3 numBlocksForTensor(const Tensor& t, dim3 blockDim);

}

namespace StrengthNetImpl
{

__device__ float& at(const Tensor& t, uint x, uint y = 0, uint z = 0); // tensor element access
__device__ float accumulate(float* data, uint n, float (*func)(float, float)); // Accumulate data with func

__global__ void scaleK(Tensor y, float w); // y = y * w
__global__ void hadamardK(Tensor y, const Tensor w); // y = y ⊙ w
__global__ void matmulK(Tensor y, const Tensor W, const Tensor x); // y = W * x
__global__ void addK(Tensor y, const Tensor x); // y = y + x
__global__ void minK(Tensor y, const Tensor x); // y = min(x) across x-dimension
__global__ void minDerivedK(Tensor x_grad, const Tensor y_grad, const Tensor x, const Tensor y); // x_grad = y_grad * d_min(x) / d_x
__global__ void sumK(Tensor y, const Tensor x); // y = sum(x) across x-dimension

__global__ void dotproduct(Tensor y, const Tensor a, const Tensor b); // y = a^T * b
__global__ void transposeMatmul(Tensor y, const Tensor a, const Tensor b, uint z_index); // y = a * b^T
__global__ void relu(Tensor h); // in-place relu
__global__ void softmax(Tensor a); // in-place softmax
__global__ void lossDerived(Tensor y_grad, float target, const Tensor y); // y_grad = d_y / d_target
__global__ void softmaxDerived(Tensor z_grad, const Tensor a); // z_grad = d_softmax(z) / d_z where a = softmax(z)
__global__ void matmulDerived(Tensor x_grad, const Tensor y_grad, const Tensor W); // x_grad = y_grad * d_y / d_x, where y = W * x
__global__ void accumulateTensorZ(Tensor W); // reduce z dimension by sum
__global__ void updateTensor(Tensor W, const Tensor W_grad, float weightPenalty, float learnrate); // W = W - W_grad * learnrate - d_(W ⊙ W) / d_W * weightPenalty

void scale(Tensor& y, float w);
void hadamard(Tensor& y, const Tensor& w);
void matmul(Tensor& y, const Tensor& W, const Tensor& x);
void add(Tensor& y, const Tensor& x);
void min(Tensor& y, const Tensor& x);
void minDerived(Tensor& x_grad, const Tensor& y_grad, const Tensor& x, const Tensor& y);
void sum(Tensor& y, const Tensor& x);

}

using namespace StrengthNetImpl;

void StrengthNet::forward() {
  // // layer 1
  matmul(h, W, x);
  b.broadcast(N, hidden_ch);
  add(h, b);
  b.broadcast(1, hidden_ch); // reset
  // matmul<<<numBlocks, blockDim2d>>>(h, W1, x);
  // numBlocks = dim3((N * h.dims.y + blockDim1d.x - 1) / blockDim1d.x);
  // relu<<<numBlocks, blockDim1d>>>(h);

  // // layer 2
  // numBlocks = numBlocksForTensor(r, blockDim2d);
  // matmul<<<numBlocks, blockDim2d>>>(r, W2r, h);
  // numBlocks = numBlocksForTensor(a, blockDim2d);
  // matmul<<<numBlocks, blockDim2d>>>(a, W2z, h);

  // // aggregate by attention
  min(y, h);
  // numBlocks = numBlocksForTensor(a, blockDim1d);
  // softmax<<<numBlocks, blockDim1d, N*sizeof(float)>>>(a);
  // dotproduct<<<1, 1>>>(y, r, a);
}

void StrengthNet::backward(float target/*, size_t index*/) {
  // assert(index < batchSize);

  target = (target - 1500.f) / 500.f;
  cudaMemcpy(tgt.data, &target, sizeof(float), cudaMemcpyHostToDevice);

  // dL/dy = 2(y - tgt)
  y_grad.assignFrom(y);
  scale(y_grad, 2.f);
  scale(tgt, -2.f);
  add(y_grad, tgt);
  // lossDerived<<<1,1>>>(y_grad, target, y); // dL/dy

  // dL/dh = dL/dy * I_min(h)
  minDerived(h_grad, y_grad, h, y);

  // dL/dW = dL/dh * x^T; dL/db = sum(dL/dh)
  x.transpose();
  matmul(W_grad, h_grad, x);
  x.transpose(); // reset
  sum(b_grad, h_grad);

  // // aggregate by attention
  // r_grad.assignFrom(a); // dy/dr
  // y_grad.broadcast(N);
  // hadamard(r_grad, y_grad); // dL/dr = dL/dy * dy/da

  // z_grad.assignFrom(r); // dy/da
  // hadamard(z_grad, y_grad); // dL/da = dL/dy * dy/da
  // softmaxDerived<<<numBlocks, blockDim1d>>>(z_grad, a); // dL/dz2 = da/dz2 * dL/da

  // // layer 2
  // numBlocks = numBlocksForTensor(W2r_grad, blockDim2d);
  // transposeMatmul<<<numBlocks, blockDim2d>>>(W2r_grad, r_grad, h, index); // dL/dW2r = dL/dr * h^T
  // transposeMatmul<<<numBlocks, blockDim2d>>>(W2z_grad, z_grad, h, index); // dL/dW2z = dL/dz * h^T

  // numBlocks = numBlocksForTensor(hr_grad, blockDim2d);
  // matmulDerived<<<numBlocks, blockDim2d>>>(hr_grad, r_grad, W2r);
  // matmulDerived<<<numBlocks, blockDim2d>>>(hz_grad, z_grad, W2z);
  // numBlocks = dim3((N * h_grad.dims.y + blockDim1d.x - 1) / blockDim1d.x);
  // add<<<numBlocks, blockDim1d>>>(h_grad, hr_grad, hz_grad); // dL/dh = dr/dh * dL/dr + dz/dh * dL/dz

  // // layer 1
  // relu<<<numBlocks, blockDim1d>>>(h_grad); // dL/dz1 = dL/dh * dh/dz1
 
  // numBlocks = numBlocksForTensor(W1_grad, blockDim2d);
  // transposeMatmul<<<numBlocks, blockDim2d>>>(W1_grad, h_grad, x, index); // dL/dW1 = dL/dz1 * x^T
}

void StrengthNet::mergeGrads() {
  accumulateTensorZ<<<1, {W_grad.dims.x, W_grad.dims.y}>>>(W_grad);
  accumulateTensorZ<<<1, {b_grad.dims.x, b_grad.dims.y}>>>(b_grad);
  W_grad.dims.z = b_grad.dims.z = 1;
  // accumulateTensorZ<<<1, {W1_grad.dims.x, W1_grad.dims.y}>>>(W1_grad);
  // accumulateTensorZ<<<1, {W2r_grad.dims.x, W2r_grad.dims.y}>>>(W2r_grad);
  // accumulateTensorZ<<<1, {W2z_grad.dims.x, W2z_grad.dims.y}>>>(W2z_grad);
  // W1_grad.dims.z = W2r_grad.dims.z = W2z_grad.dims.z = 1;
}

void StrengthNet::update(float weightPenalty, float learnrate) {
  updateTensor<<<1, {W.dims.x, W.dims.y}>>>(W, W_grad, weightPenalty, learnrate);
  updateTensor<<<1, {b.dims.x, b.dims.y}>>>(b, b_grad, weightPenalty, learnrate);
  // updateTensor<<<1, {W1.dims.x, W1.dims.y}>>>(W1, W1_grad, weightPenalty, learnrate);
  // updateTensor<<<1, {W2r.dims.x, W2r.dims.y}>>>(W2r, W2r_grad, weightPenalty, learnrate);
  // updateTensor<<<1, {W2z.dims.x, W2z.dims.y}>>>(W2z, W2z_grad, weightPenalty, learnrate);
}

namespace {

constexpr dim3 numBlocksForTensor(const Tensor& t, dim3 blockDim) {
  dim3 numBlocks(1, 1, 1);
  numBlocks.x = (t.dims.x + blockDim.x - 1) / blockDim.x;
  numBlocks.y = (t.dims.y + blockDim.y - 1) / blockDim.y;
  return numBlocks;
}

}

namespace StrengthNetImpl
{

__device__ float& at(const Tensor& t, uint x, uint y, uint z) {
  assert(x < t.viewDims.x);
  assert(y < t.viewDims.y);
  assert(z < t.viewDims.z);

  // implement broadcast
  if(1 == t.dims.x)
    x = 0;
  if(1 == t.dims.y)
    y = 0;
  if(1 == t.dims.z)
    z = 0;

  uint xy;

  // implement transposition
  if(t.transposed)
    xy = y * t.dims.x + x;
  else
    xy = x * t.dims.y + y;
  
  return t.data[z * t.dims.x * t.dims.y + xy];
}

__device__ float accumulate(float* data, uint n, float (*func)(float, float)) {
  assert(0 == blockIdx.x); // this operation cannot use blocks
  assert(blockDim.x >= n); // must have enough threads
  extern __shared__ float buffer[];
  uint i = threadIdx.x;

  if(i < n)
    buffer[i] = data[i];

  __syncthreads();

  for (uint s = n/2; s > 0; s = n/2) {
      if (i + n - s < n) { // accumulate second half of data into first half
          buffer[i] = func(buffer[i], buffer[i + n - s]);
      }
      n -= s; // discard second half
      __syncthreads();
  }

  // now we are down to exactly either zero, one or two elements
  if(2 == n)
    return func(buffer[0], buffer[1]);
  else if(1 == n)
    return buffer[0];
  else
    return NAN; // some default
}

__global__ void scaleK(Tensor y, float w) {
  uint xx = blockIdx.x * blockDim.x + threadIdx.x;
  uint yy = blockIdx.y * blockDim.y + threadIdx.y;
  uint zz = blockIdx.z * blockDim.z + threadIdx.z;
  if(xx < y.dims.x && yy < y.dims.y && zz < y.dims.z)
    at(y, xx, yy, zz) *= w;
}

__global__ void hadamardK(Tensor y, const Tensor w) {
  assert(y.dims.x == w.viewDims.x);
  assert(y.dims.y == w.viewDims.y);
  assert(y.dims.z == w.viewDims.z);
  uint xx = blockIdx.x * blockDim.x + threadIdx.x;
  uint yy = blockIdx.y * blockDim.y + threadIdx.y;
  uint zz = blockIdx.z * blockDim.z + threadIdx.z;
  if(xx < y.dims.x && yy < y.dims.y && zz < y.dims.z)
    at(y, xx, yy, zz) *= at(w, xx, yy, zz);
}

// y = W*x
// set blocks to partition y into squares
__global__ void matmulK(Tensor y, const Tensor W, const Tensor x) {
  assert(W.dims.x == x.dims.y); // input size must match
  assert(y.dims.x == x.dims.x); // output size must match
  assert(y.dims.y == W.dims.y); // output size must match
  assert(1 == W.dims.z);
  assert(1 == x.dims.z); // TODO: support parallelism

  // naive implementation
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  uint col = blockIdx.x * blockDim.x + threadIdx.x;

  // early exit for overspilling blocks
  if(row >= y.dims.y || col >= y.dims.x)
    return;

  float h = 0.0f;
  for (uint i = 0; i < x.dims.y; i++) {
    h += at(W, i, row) * at(x, col, i);
  }
  at(y, col, row) = h;
}

__global__ void addK(Tensor y, const Tensor x) {
  assert(y.dims.x == x.viewDims.x);
  assert(y.dims.y == x.viewDims.y);
  assert(y.dims.z == x.viewDims.z);

  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= y.dims.x * y.dims.y)
    return;

  y.data[i] += x.data[i];
}

__global__ void minK(Tensor y, const Tensor x) {
  assert(1 == y.dims.x);
  assert(y.dims.y == x.viewDims.y);
  assert(y.dims.z == x.viewDims.z);
  assert(1 == y.dims.y); // TODO: parallelism support
  assert(1 == y.dims.z); // TODO: parallelism support

  // uint yy = blockIdx.y * blockDim.y + threadIdx.y;
  // uint zz = blockIdx.z * blockDim.z + threadIdx.z;
  // if(yy >= y.dims.y || zz >= y.dims.z)
  //   return;

  float minValue = accumulate(x.data, x.dims.x, [](float a, float b) { return a < b ? a : b; });
  if(0 == threadIdx.x)
    y.data[0] = minValue;
}

__global__ void minDerivedK(Tensor x_grad, const Tensor y_grad, const Tensor x, const Tensor y) {
  assert(x_grad.dims.x == x.viewDims.x);
  assert(x_grad.dims.y == x.viewDims.y);
  assert(x_grad.dims.z == x.viewDims.z);
  assert(y_grad.viewDims.x == y.viewDims.x);
  assert(y_grad.viewDims.y == y.viewDims.y);
  assert(y_grad.viewDims.z == y.viewDims.z);
  assert(1 == y.viewDims.x);
  assert(y.viewDims.y == x.viewDims.y);
  assert(y.viewDims.z == x.viewDims.z);
  assert(1 == y.viewDims.y); // TODO: parallelism support
  assert(1 == y.viewDims.z); // TODO: parallelism support
  assert(0 == blockIdx.x); // TODO: parallelism support

  if(threadIdx.x >= x_grad.dims.x)
    return;

  float minValue = y.data[0];
  x_grad.data[threadIdx.x] = y_grad.data[0] * (x.data[threadIdx.x] == minValue);
}

__global__ void sumK(Tensor y, const Tensor x) {
  assert(1 == y.dims.x);
  assert(y.dims.y == x.viewDims.y);
  assert(y.dims.z == x.viewDims.z);
  assert(1 == y.dims.y); // TODO: parallelism support
  assert(1 == y.dims.z); // TODO: parallelism support

  float sumValue = accumulate(x.data, x.viewDims.x, [](float a, float b) { return a + b; });
  if(0 == threadIdx.x)
    y.data[0] = sumValue;
}

__global__ void dotproduct(Tensor y, const Tensor a, const Tensor b) {
  assert(a.dims.x == b.dims.x);

  float s = 0;
  for(int i = 0; i < a.dims.x; i++)
    s += at(a, i) * at(b, i);
  at(y, 0) = s;
}

// y[z_index] = a*b^T
// set blocks to partition y into squares
__global__ void transposeMatmul(Tensor y, const Tensor a, const Tensor b, uint z_index) {
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

  float h = 0.0f;
  for (int i = 0; i < b.dims.x; i++) {
    if(col < b.dims.y)
      h += at(a, i, row) * at(b, i, col);
    else // construct bias row (as if b.data[...] == 1)
      h += at(a, i, row);
  }
  at(y, col, row, z_index) = h;
}

__global__ void relu(Tensor h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= h.dims.x * h.dims.y)
    return;

  if(h.data[i] < 0)
    h.data[i] = 0;
}

// uses a.dims.x shared memory floats
__global__ void softmax(Tensor a) {
  assert(1 == a.dims.y);
  assert(1 == a.dims.z); // TODO: batch softmax
  assert(a.dims.x == a.viewDims.x);

  int i = threadIdx.x;
  if(i >= a.dims.x)
    return;

  float max_a = accumulate(a.data, a.dims.x, [](float a, float b) { return a > b ? a : b; });
  at(a, i) = expf(at(a, i) - max_a);  // -max(a) improves numerical stability without changing the result
  float sum_a = accumulate(a.data, a.dims.x, [](float a, float b) { return a + b; });
  at(a, i) /= sum_a;
}

__global__ void lossDerived(Tensor y_grad, float target, const Tensor y) {
  at(y_grad, 0) = 2.f * (at(y, 0) - target);
}

__global__ void softmaxDerived(Tensor z_grad, const Tensor a) {
  assert(z_grad.dims.x == a.dims.x);

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j >= z_grad.dims.x)
    return;

  float a_j = at(a, j);
  float b = 0;
  for(uint i = 0; i < a.dims.x; i++) {
    float delta = i == threadIdx.x ? 1.f : 0.f;
    b += at(z_grad, i) * at(a, i) * (delta - a_j);
  }
  __syncthreads();
  at(z_grad, j) = b;
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

  float h = 0.0f;
  for (int i = 0; i < y_grad.dims.y; i++) {
    h += at(W, row, i) * at(y_grad, col, i);
  }
  at(x_grad, col, row) = h;
}

// 1 block with W.dims threads
__global__ void accumulateTensorZ(Tensor W) {
  float v = 0;
  for(uint z = 0; z < W.dims.z; z++) {
    v += at(W, threadIdx.x, threadIdx.y, z);
  }
  at(W, threadIdx.x, threadIdx.y, 0) = v;
}

// 1 block with W.dims threads
__global__ void updateTensor(Tensor W, const Tensor W_grad, float weightPenalty, float learnrate) {
  assert(W.dims.x == W_grad.dims.x);
  assert(W.dims.y == W_grad.dims.y);
  assert(1 == W_grad.dims.z);

  float delta = 0;
  for(uint z = 0; z < W_grad.dims.z; z++) {
    delta += at(W_grad, threadIdx.x, threadIdx.y, z);
  }
  delta += at(W, threadIdx.x, threadIdx.y) * 2 * weightPenalty;
  at(W, threadIdx.x, threadIdx.y) -= delta * learnrate;
}

// __global__ void forwardTanhKernel(float* softx, int ch, int row) {
//   softx[threadIdx.x * ch + row] = 10.f * tanhf(softx[threadIdx.x * ch + row]);
// }

// __global__ void backwardTanhKernel(float* ingrads, float* outputx, float* outgrads) {
//   float cosa = cosf(outputx[threadIdx.x*2 + 1]);
//   outgrads[threadIdx.x*2 + 1] = ingrads[threadIdx.x*2 + 1] * 10.f / (cosa*cosa);
// }

void scale(Tensor& y, float w) {
  dim3 blockDim(16, 16, 4);
  dim3 numBlocks = numBlocksForTensor(y, blockDim);
  scaleK<<<numBlocks, blockDim>>>(y, w);
}

void hadamard(Tensor& y, const Tensor& w) {
  dim3 blockDim(16, 16, 4);
  dim3 numBlocks = numBlocksForTensor(y, blockDim);
  hadamardK<<<numBlocks, blockDim>>>(y, w);
}

void matmul(Tensor& y, const Tensor& W, const Tensor& x) {
  dim3 blockDim(16, 16);
  dim3 numBlocks = numBlocksForTensor(y, blockDim);
  matmulK<<<numBlocks, blockDim>>>(y, W, x);
}

void add(Tensor& y, const Tensor& x) {
  uint numBlocks = (y.dims.x * y.dims.y * y.dims.z + 1023) / 1024;
  addK<<<numBlocks, 1024>>>(y, x);
}

void min(Tensor& y, const Tensor& x) {
  uint numBlocks = (x.dims.x + 1023) / 1024;
  minK<<<numBlocks, 1024, x.dims.x*sizeof(float)>>>(y, x);
}

void minDerived(Tensor& x_grad, const Tensor& y_grad, const Tensor& x, const Tensor& y) {
  uint numBlocks = (x.dims.x + 1023) / 1024;
  minDerivedK<<<numBlocks, 1024, x.dims.x*sizeof(float)>>>(x_grad, y_grad, x, y);
}

void sum(Tensor& y, const Tensor& x) {
  uint numBlocks = (x.dims.x + 1023) / 1024;
  sumK<<<numBlocks, 1024, x.dims.x*sizeof(float)>>>(y, x);
}

} // end namespace StrengthNetImpl
