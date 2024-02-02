#include "strengthnet.h"
#include <iostream>
#include <vector>
#include <cmath>

namespace {

constexpr dim3 numBlocksForTensor(const Tensor& t, dim3 blockDim);

}

namespace StrengthNetImpl
{

__device__ void getxdim(const Tensor& t, uint z, uint& xdim, uint& offset); // get x-size and offset of z'th layer
__device__ float& at(const Tensor& t, uint x, uint y = 0, uint z = 0); // tensor element access

__global__ void atK(Tensor t, float* out, uint x, uint y, uint z);
__global__ void scaleK(Tensor y, float w); // y = y * w
__global__ void hadamardK(Tensor y, const Tensor w); // y = y ⊙ w
__global__ void matmulK(Tensor y, const Tensor W, const Tensor x); // y = W * x
__global__ void addK(Tensor y, const Tensor x); // y = y + x
__host__ __global__ enum { OP_PLUS, OP_MIN }; // for use with accumulateK
__global__ void accumulateK(Tensor a, const Tensor t, uint op = OP_PLUS); // a = op(t) across x-dimension
__global__ void minDerivedK(Tensor x_grad, const Tensor y_grad, const Tensor x, const Tensor y); // x_grad = y_grad * d_min(x) / d_x

__global__ void reluK(Tensor h); // in-place relu
__global__ void reluDerivedK(Tensor g, const Tensor a);  // in-place g *= d_relu(a) / d_a
__global__ void softmaxK(Tensor a); // in-place softmax
__global__ void softmaxDerivedK(Tensor a_grad, const Tensor z_grad, const Tensor a); // a_grad = z_grad * d_softmax(z) / d_z where a = softmax(z)

__global__ void updateTensorK(Tensor W, const Tensor W_grad, float weightPenalty, float learnrate); // W = W - W_grad * learnrate - d_(W ⊙ W) / d_W * weightPenalty

float getelem(const Tensor &t, uint x, uint y, uint z); // tensor element access (slow/wasteful)
void scale(Tensor& y, float w);
void hadamard(Tensor& y, const Tensor& w);
void matmul(Tensor& y, const Tensor& W, const Tensor& x);
void add(Tensor& y, const Tensor& x);
void relu(Tensor& t);
void reluDerived(Tensor& g, const Tensor& a);
void min(Tensor& y, const Tensor& x);
void max(Tensor& y, const Tensor& x);
void minDerived(Tensor& x_grad, const Tensor& y_grad, const Tensor& x, const Tensor& y);
void sum(Tensor& y, const Tensor& x);
void softmax(Tensor& a);
void softmaxDerived(Tensor& a_grad, const Tensor& z_grad, const Tensor& a);

}

using namespace StrengthNetImpl;

void StrengthNet::forward() {
  assert(x); // tensors must be allocated by previous setInput()

  // layer 1
  h->cat(); x->cat();
  matmul(*h, W1, *x);
  b1.broadcast(N, hidden_ch);
  add(*h, b1);
  b1.broadcast(1, hidden_ch); // reset
  x->uncat();
  // h->uncat();
  // min(*y, *h);
  relu(*h);

  // layer 2
  r->cat();
  matmul(*r, W2r, *h);
  b2r.broadcast(N, 1);
  add(*r, b2r);
  b2r.broadcast(1, 1); // reset
  r->uncat();
  a->cat();
  matmul(*a, W2z, *h);
  b2z.broadcast(N, 1);
  add(*a, b2z);
  b2z.broadcast(1, 1); // reset
  a->uncat();
  h->uncat();

  // aggregate by attention
  softmax(*a);
  ra->assignFrom(*r);
  hadamard(*ra, *a);
  sum(*y, *ra);
}

void StrengthNet::backward() {
  assert(tgt); // tensors must be allocated by previous setInput()

  // dL/dy = 2(y - tgt)
  y_grad->assignFrom(*y);
  scale(*y_grad, -1.f);
  add(*y_grad, *tgt);
  scale(*y_grad, -2.f);

  // // dL/dh = dL/dy * I_min(h)
  // minDerived(*h_grad, *y_grad, *h, *y);

  // dL/dr = dL/dy * a
  r_grad->assignFrom(*a); // dy/dr
  uint batchSize = zoffset.size()-1;
  y_grad->broadcast(N, 1, batchSize);
  hadamard(*r_grad, *y_grad);

  a_grad->assignFrom(*r);
  hadamard(*a_grad, *y_grad);
  y_grad->broadcast(batchSize, 1, batchSize); // reset
  softmaxDerived(*z_grad, *a_grad, *a); // dy/da

  // layer 2
  h->transpose(); h->cat();
  r_grad->cat(); z_grad->cat();

  matmul(*W2r_grad, *r_grad, *h);
  sum(*b2r_grad, *r_grad);

  W2r.transpose(); hr_grad->cat();
  matmul(*hr_grad, W2r, *r_grad);
  W2r.transpose(); hr_grad->uncat();
  
  matmul(*W2z_grad, *z_grad, *h);
  sum(*b2z_grad, *z_grad);

  W2z.transpose(); hz_grad->cat();
  matmul(*hz_grad, W2z, *z_grad);
  W2z.transpose(); hz_grad->uncat();

  r_grad->uncat(); z_grad->uncat();
  h->transpose(); h->uncat(); // reset

  h_grad->assignFrom(*hr_grad);
  add(*h_grad, *hz_grad);

  // layer 1
  reluDerived(*h_grad, *h);

  // dL/dW = dL/dh * x^T; dL/db = sum(dL/dh)
  x->transpose();
  h_grad->cat(); x->cat();
  matmul(*W1_grad, *h_grad, *x);
  sum(*b1_grad, *h_grad);
  h_grad->uncat(); x->uncat();
  x->transpose(); // reset
}

void StrengthNet::update(float weightPenalty, float learnrate) {
  updateTensorK<<<1, {W1.dims.x, W1.dims.y}>>>(W1, *W1_grad, weightPenalty, learnrate);
  updateTensorK<<<1, {b1.dims.x, b1.dims.y}>>>(b1, *b1_grad, weightPenalty, learnrate);
  updateTensorK<<<1, {W2r.dims.x, W2r.dims.y}>>>(W2r, *W2r_grad, weightPenalty, learnrate);
  updateTensorK<<<1, {b2r.dims.x, b2r.dims.y}>>>(b2r, *b2r_grad, weightPenalty, learnrate);
  updateTensorK<<<1, {W2z.dims.x, W2z.dims.y}>>>(W2z, *W2z_grad, weightPenalty, learnrate);
  updateTensorK<<<1, {b2z.dims.x, b2z.dims.y}>>>(b2z, *b2z_grad, weightPenalty, learnrate);
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

__device__ void getxdim(const Tensor& t, uint z, uint& xdim, uint& offset) {
  assert(z < t.viewDims.z);
  if(1 == t.dims.z) // broadcast
    z = 0;
  offset = t.zoffset[z];
  uint upper = t.zoffset[z+1];
  if(1 == t.viewDims.z) // cat/unbatching
    upper = t.transposed ? t.viewDims.y : t.viewDims.x;
  xdim = upper - offset;
}

__device__ float& at(const Tensor& t, uint x, uint y, uint z) {
  // un-transpose
  uint dimx = t.transposed ? t.dims.y : t.dims.x;
  uint dimy = t.transposed ? t.dims.x : t.dims.y;
  uint vdimy = t.transposed ? t.viewDims.x : t.viewDims.y;
  if(t.transposed) {
    float tmp = x;
    x = y;
    y = tmp;
  }
  assert(y < vdimy);

  uint xdim, offset; // dimx covers whole batch, xdim covers z'th layer
  getxdim(t, z, xdim, offset);

  // implement broadcast
  if(t.dims.z == dimx)
    x = 0;
  if(1 == dimy)
    y = 0;
  assert(x < xdim);

  return t.data[(offset + x) * dimy + y];
}

__global__ void atK(Tensor t, float* out, uint x, uint y = 0, uint z = 0) {
  *out = at(t, x, y, z);
}

__global__ void scaleK(Tensor y, float w) {
  uint xx = blockIdx.x * blockDim.x + threadIdx.x;
  if(xx < y.dims.x * y.dims.y)
    y.data[xx] *= w;
}

__global__ void hadamardK(Tensor y, Tensor w) {
  assert(y.dims.x == w.viewDims.x);
  assert(y.dims.y == w.viewDims.y);
  assert(y.dims.z == w.viewDims.z);

  uint xx = blockIdx.x * blockDim.x + threadIdx.x;
  uint yy = blockIdx.y * blockDim.y + threadIdx.y;
  uint zz = blockIdx.z * blockDim.z + threadIdx.z;

  if(yy >= y.dims.y || zz >= y.dims.z)
    return;

  uint xdim, offset; // dimx covers whole batch, xdim covers z'th layer
  getxdim(y, zz, xdim, offset);

  if(xx >= xdim)
    return;

  at(y, xx, yy, zz) *= at(w, xx, yy, zz);
}

// y = W*x
// set blocks to partition y into squares
__global__ void matmulK(Tensor y, const Tensor W, const Tensor x) {
  assert(W.viewDims.x == x.viewDims.y); // input size must match
  assert(y.dims.x == x.viewDims.x); // output size must match
  assert(y.dims.y == W.viewDims.y); // output size must match
  assert(1 == W.viewDims.z); // tensors must be cat() for matmul
  assert(1 == x.viewDims.z);
  assert(1 == y.viewDims.z);
  assert(!y.transposed); // view on output must be unchanged to ensure proper bounds

  // naive implementation
  uint row = blockIdx.y * blockDim.y + threadIdx.y;
  uint col = blockIdx.x * blockDim.x + threadIdx.x;

  // early exit for overspilling blocks
  if(row >= y.dims.y || col >= y.dims.x)
    return;

  float h = 0.0f;
  for (uint i = 0; i < x.viewDims.y; i++) {
    h += at(W, i, row) * at(x, col, i);
  }
  at(y, col, row) = h;
}

__global__ void addK(Tensor y, const Tensor x) {
  assert(y.dims.x == x.viewDims.x);
  assert(y.dims.y == x.viewDims.y);
  assert(y.viewDims.z == x.viewDims.z);
  assert(y.viewDims.z <= y.dims.z); // do not multiple-assign

  uint xx = blockIdx.x * blockDim.x + threadIdx.x;
  uint yy = blockIdx.y * blockDim.y + threadIdx.y;
  uint zz = blockIdx.z * blockDim.z + threadIdx.z;

  if(yy >= y.dims.y || zz >= y.viewDims.z)
    return;

  uint xdim, offset; // dimx covers whole batch, xdim covers z'th layer
  getxdim(y, zz, xdim, offset);

  if(xx >= xdim)
    return;

  at(y, xx, yy, zz) += at(x, xx, yy, zz);
}

__global__ void accumulateK(Tensor a, const Tensor t, uint op) {
  assert(a.dims.y == t.viewDims.y);
  assert(a.dims.z == t.viewDims.z);
  if(0 == t.viewDims.x) // empty tensor: nothing to do
    return;
  assert(0 == blockIdx.x);
  assert(0 == threadIdx.x); // x-sum is single-threaded

  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint z = blockIdx.z * blockDim.z + threadIdx.z;

  if(y >= t.dims.y || z >= t.viewDims.z)
    return;

  uint xdim, offset;
  getxdim(t, z, xdim, offset);

  float v = 0; // some default value; the case xdim==0 must be handled!
  if(xdim > 0)
    v = at(t, 0, y, z);
  for(uint x = 1; x < xdim; x++) {
    float elem = at(t, x, y, z);
    switch(op) {
      default:
      case OP_PLUS: v = v + elem; break;
      case OP_MIN: if(elem < v) v = elem; break;
    }
  }
  at(a, 0, y, z) = v;
}

__global__ void minDerivedK(Tensor x_grad, const Tensor y_grad, const Tensor x, const Tensor y) {
  assert(x_grad.dims.x == x.viewDims.x);
  assert(x_grad.dims.y == x.viewDims.y);
  assert(x_grad.dims.z == x.viewDims.z);
  assert(y_grad.viewDims.x == y.viewDims.x);
  assert(y_grad.viewDims.y == y.viewDims.y);
  assert(y_grad.viewDims.z == y.viewDims.z);
  assert(y.viewDims.z == y.viewDims.x); // there can only be 1 minimum value
  assert(y.viewDims.y == x.viewDims.y);
  assert(y.viewDims.z == x.viewDims.z);
  assert(1 == y.viewDims.y); // no y parallelism support

  uint xx = blockIdx.x * blockDim.x + threadIdx.x;
  uint zz = blockIdx.z * blockDim.z + threadIdx.z;

  if(zz >= y.dims.z)
    return;

  uint xdim, offset;
  getxdim(x, zz, xdim, offset);

  if(xx >= xdim)
    return;

  float minValue = at(y, 0, 0, zz);
  at(x_grad, xx, 0, zz) = at(y_grad, 0, 0, zz) * (at(x, xx, 0, zz) <= minValue);
}

__global__ void reluK(Tensor h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= h.dims.x * h.dims.y)
    return;

  if(h.data[i] < 0)
    h.data[i] = 0;
}

__global__ void reluDerivedK(Tensor g, const Tensor a) {
  assert(g.dims.x == a.dims.x);
  assert(g.dims.y == a.dims.y);
  assert(g.dims.z == a.dims.z);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= g.dims.x * g.dims.y)
    return;

  if(a.data[i] <= 0)
    g.data[i] = 0;
}

__global__ void softmaxK(Tensor a) {
  assert(1 == a.dims.y);
  assert(a.dims.x == a.viewDims.x);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= a.dims.z)
    return;

  uint xdim, offset;
  getxdim(a, i, xdim, offset);

  float maxValue = at(a, 0, 0, i);
  for(uint j = 1; j < xdim; j++) {
    float v = at(a, j, 0, i);
    if(v > maxValue)
      maxValue = v;
  }

  float sum = 0;
  for(uint j = 0; j < xdim; j++) {
    float v = exp(at(a, j, 0, i));
    at(a, j, 0, i) = v;
    sum += v;
  }

  for(uint j = 0; j < xdim; j++) {
    at(a, j, 0, i) = at(a, j, 0, i) / sum;
  }
}

__global__ void softmaxDerivedK(Tensor a_grad, const Tensor z_grad, const Tensor a) {
  assert(a_grad.dims.x == a.dims.x);
  assert(a_grad.dims.y == a.dims.y);
  assert(z_grad.dims.x == a.dims.x);
  assert(1 == z_grad.dims.y);
  assert(1 == a.dims.y);
  assert(z_grad.dims.y == a.dims.y);

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if(j >= z_grad.dims.x)
    return;

  // find my batch layer according to my index j
  uint zj = j * a.dims.z / a.dims.x; // should be pretty close, now adjust step by step
  while(a.zoffset[zj] > j)
    zj--;
  while(a.zoffset[zj+1] <= j)
    zj++;
  uint lower = a.zoffset[zj];
  uint upper = a.zoffset[zj+1];

  float a_j = a.data[j];
  float b = 0;
  for(uint i = lower; i < upper; i++) {
    float delta = i == j ? 1.f : 0.f;
    b += z_grad.data[i] * a.data[i] * (delta - a_j);
  }
  a_grad.data[j] = b;
}

// 1 block with W.dims threads
__global__ void updateTensorK(Tensor W, const Tensor W_grad, float weightPenalty, float learnrate) {
  assert(W.dims.x == W_grad.dims.x);
  assert(W.dims.y == W_grad.dims.y);
  assert(1 == W_grad.viewDims.z);

  float delta = at(W_grad, threadIdx.x, threadIdx.y) + at(W, threadIdx.x, threadIdx.y) * 2 * weightPenalty;
  at(W, threadIdx.x, threadIdx.y) -= delta * learnrate;
}

// __global__ void forwardTanhKernel(float* softx, int ch, int row) {
//   softx[threadIdx.x * ch + row] = 10.f * tanhf(softx[threadIdx.x * ch + row]);
// }

// __global__ void backwardTanhKernel(float* ingrads, float* outputx, float* outgrads) {
//   float cosa = cosf(outputx[threadIdx.x*2 + 1]);
//   outgrads[threadIdx.x*2 + 1] = ingrads[threadIdx.x*2 + 1] * 10.f / (cosa*cosa);
// }

float getelem(const Tensor &t, uint x, uint y, uint z) {
  float* elem;
  cudaMalloc(&elem, sizeof(float));
  atK<<<1, 1>>>(t, elem, x, y, z);
  float out;
  cudaMemcpy(&out, elem, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(elem);
  return out;
}

void scale(Tensor& y, float w) {
  uint numBlocks = (y.dims.x*y.dims.y + 1023) / 1024;
  scaleK<<<numBlocks, 1024>>>(y, w);
}

void hadamard(Tensor& y, const Tensor& w) {
  dim3 blockDim(8, 8, 8);
  dim3 numBlocks = numBlocksForTensor(y, blockDim);
  numBlocks.z = (y.dims.z + 7) / blockDim.z;
  hadamardK<<<numBlocks, blockDim>>>(y, w);
}

void matmul(Tensor& y, const Tensor& W, const Tensor& x) {
  dim3 blockDim(16, 16, 1);
  dim3 numBlocks = numBlocksForTensor(y, blockDim);
  matmulK<<<numBlocks, blockDim>>>(y, W, x);
}

void add(Tensor& y, const Tensor& x) {
  dim3 blockDim(16, 16, 4);
  dim3 numBlocks = numBlocksForTensor(y, blockDim);
  addK<<<numBlocks, blockDim>>>(y, x);
}

void relu(Tensor& t) {
  uint numBlocks = (t.dims.x * t.dims.y + 1023) / 1024;
  reluK<<<numBlocks, 1024>>>(t);
}

void reluDerived(Tensor& g, const Tensor& a) {
  uint numBlocks = (g.dims.x * g.dims.y + 1023) / 1024;
  reluDerivedK<<<numBlocks, 1024>>>(g, a);
}

void min(Tensor& y, const Tensor& x) {
  dim3 blockDim(1, 32, 32);
  dim3 numBlocks(1, (x.viewDims.y+31)/32, (x.viewDims.z+31)/32);
  accumulateK<<<numBlocks, blockDim>>>(y, x, OP_MIN);
}

void minDerived(Tensor& x_grad, const Tensor& y_grad, const Tensor& x, const Tensor& y) {
  dim3 blockDim(32, 1, 32);
  dim3 numBlocks((x.dims.x + 31) / 32, 1, (x.dims.z + 31) / 32);
  minDerivedK<<<numBlocks, blockDim>>>(x_grad, y_grad, x, y);
}

void sum(Tensor& y, const Tensor& x) {
  dim3 blockDim(1, 32, 32);
  dim3 numBlocks(1, (x.viewDims.y+31)/32, (x.viewDims.z+31)/32);
  accumulateK<<<numBlocks, blockDim>>>(y, x, OP_PLUS);
}

void softmax(Tensor& a) {
  uint numBlocks = (a.dims.z + 1023) / 1024;
  softmaxK<<<numBlocks, 1024>>>(a);
}

void softmaxDerived(Tensor& a_grad, const Tensor& z_grad, const Tensor& a) {
  uint numBlocks = (a.dims.x + 1023) / 1024;
  softmaxDerivedK<<<numBlocks, 1024>>>(a_grad, z_grad, a);
}

} // end namespace StrengthNetImpl
