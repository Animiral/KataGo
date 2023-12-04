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
      if (threadIdx.x + s < n) {
          buffer[threadIdx.x] = max(buffer[threadIdx.x], buffer[threadIdx.x + s]);
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

  // assert(input.dims.x * dims.y <= maxN);
  // assert(reinterpret_cast<const float*>(&input[0]) + in_ch == reinterpret_cast<const float*>(&input[1])); // crude packing/alignment safety check
  // N = input.size();
  // const float* inputPtr = reinterpret_cast<const float*>(input.data());
  // CUDA_ERR("StrengthNet.forward", cudaMemcpy(inputx, inputPtr, N * in_ch * sizeof(float), cudaMemcpyHostToDevice));

  // forwardFull(inputx, hiddenx, Wh, bh, N, in_ch, hidden_ch);
  // forwardRelu(hiddenx, N, hidden_ch);
  // forwardFull(hiddenx, outputx, Wo, bo, N, hidden_ch, out_ch);
  // forwardAggregate(outputx, softx, aggregx, N, out_ch);

  // StrengthNet::Output result;
  // CUDA_ERR("StrengthNet.forward", cudaMemcpy(&result, aggregx, sizeof(StrengthNet::Output), cudaMemcpyDeviceToHost));
  // return result;
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

  // CUDA_ERR("StrengthNet.backward", cudaMemcpy(back1x, &target, sizeof(StrengthNet::Output), cudaMemcpyHostToDevice));
  // // dumpDeviceArray("inputx", inputx, N, in_ch);
  // // dumpDeviceArray("hiddenx", hiddenx, N, hidden_ch);
  // // dumpDeviceArray("outputx", outputx, N, out_ch);
  // // dumpDeviceArray("softx", softx, N, out_ch);
  // // dumpDeviceArray("aggregx", aggregx, 1, 1);
  // backwardLoss(back1x, aggregx, back2x);
  // dumpDeviceArray("dLoss/dAggreg", back2x, 1, 1);

  // backwardAggregate(back2x, softx, N, back1x);
  // backwardTanh(back1x, outputx, N, back2x);
  // dumpDeviceArray("dLoss/dOutput", back2x, N, out_ch);
  // backwardFull(back2x, hiddenx, Wo, bo, hidden_ch, out_ch, N, learnrate, back1x);
  // dumpDeviceArray("dLoss/dHidden", back1x, N, hidden_ch);
  // backwardRelu(back1x, hiddenx, hidden_ch, N, back2x);
  // dumpDeviceArray("dLoss/dHidden+ReLu", back2x, N, hidden_ch);
  // backwardFull(back2x, inputx, Wh, bh, in_ch, hidden_ch, N, learnrate, back1x);
  // dumpDeviceArray("Wo", Wo, hidden_ch, out_ch);
  // dumpDeviceArray("bo", bo, 1, out_ch);
  // dumpDeviceArray("Wh", Wh, in_ch, hidden_ch);
  // dumpDeviceArray("bh", bh, 1, hidden_ch);
}


// unit tests for local defined kernel functions

#include <sstream>
// static void checkCudaError(const cudaError_t status, const char* opName, const char* file, const char* func, int line) {
//   if(status != cudaSuccess)
//     cout << std::string("CUDA Error, for ") << opName << " file " << file << ", func " << func << ", line " << line << ", error " << cudaGetErrorString(status);
// }
// #define CUDA_ERR(opName,x) { checkCudaError((x),opName,__FILE__,#x,__LINE__); }

using namespace std;

namespace Tests {
void runStrengthModelTests() {
  cout << "Running strength model tests" << endl;
  ostringstream out;

  {
    cout << "- matmul: ";
    vector<float> A_data = {1, -3, 3, 2, -2, -1, 1, 0, -1}; // left operand, 3x3 (with bias column)
    vector<float> B_data = {7, -10, 3, 8, -11, 4, 8, 7}; // right operand, 2x4
    vector<float> C_data = {-12, -1, 30,  20, -25, 0,  -2, 25, -38,  23, -38, 16}; // expected result, 3x4
    Tensor A(3, 3);
    Tensor B(4, 2);
    Tensor C(4, 3);
    cudaMemcpy(A.data, A_data.data(), 3*3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data, B_data.data(), 4*2 * sizeof(float), cudaMemcpyHostToDevice);

    matmul<<<1, {16, 16}>>>(C, A, B);
    vector<float> C_result = static_cast<vector<float>>(C);
    bool pass = true;
    for(size_t i = 0; i < C_data.size(); i++)
      if(fabs(C_data[i] - C_result[i]) > 0.0001)
        pass = false;

    cout << (pass ? "pass" : "fail") << "\n";

    if(!pass) {
      Tensor C_expected(4,3);
      cudaMemcpy(C_expected.data, C_data.data(), 4*3 * sizeof(float), cudaMemcpyHostToDevice);
      C_expected.print(std::cout, "matmul expected");
      C.print(std::cout, "matmul result");
    }
  }

  {
    cout << "- transposeMatmul: ";
    vector<float> A_data = {1, -3, 3, 2, -2, -1}; // left operand, 3x2
    vector<float> B_data = {7, 3, -11, 8, -10, 8, 4, 7}; // right operand, 4x2
    vector<float> C_data = {-13, -1, 31,  19, -25, 1,  -3, 25, -37,  22, -38, 17}; // expected result, 3x4
    Tensor A(2, 3);
    Tensor B(2, 4);
    Tensor C(4, 3);
    cudaMemcpy(A.data, A_data.data(), 2*3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data, B_data.data(), 4*2 * sizeof(float), cudaMemcpyHostToDevice);

    transposeMatmul<<<1, {16, 16}>>>(C, A, B);
    vector<float> C_result = static_cast<vector<float>>(C);
    bool pass = true;
    for(size_t i = 0; i < C_data.size(); i++)
      if(fabs(C_data[i] - C_result[i]) > 0.0001)
        pass = false;

    cout << (pass ? "pass" : "fail") << "\n";

    if(!pass) {
      Tensor C_expected(4,3);
      cudaMemcpy(C_expected.data, C_data.data(), 4*3 * sizeof(float), cudaMemcpyHostToDevice);
      C_expected.print(std::cout, "transposeMatmul expected");
      C.print(std::cout, "transposeMatmul result");
    }
  }

  {
    cout << "- softmax: ";
    vector<float> A_data = {2.796027196f, 3.306852819f, 2.390562088f}; // operand, 3x1
    vector<float> S_data = {.3f, .5f, .2f}; // expected result, 3x1
    Tensor A(3, 1);
    cudaMemcpy(A.data, A_data.data(), 3*1 * sizeof(float), cudaMemcpyHostToDevice);

    // softmax<<<1, 3>>>(A);
    softmax<<<1, 3, 3*sizeof(float)>>>(A);
    vector<float> S_result = static_cast<vector<float>>(A);
    bool pass = true;
    for(size_t i = 0; i < S_data.size(); i++)
      if(fabs(S_data[i] - S_result[i]) > 0.0001)
        pass = false;

    cout << (pass ? "pass" : "fail") << "\n";

    if(!pass) {
      Tensor A_expected(3,1);
      cudaMemcpy(A_expected.data, S_data.data(), 3*1 * sizeof(float), cudaMemcpyHostToDevice);
      A_expected.print(std::cout, "softmax expected");
      A.print(std::cout, "softmax result");
    }
  }

  {
    cout << "- matmulDerived: ";
    vector<float> W_data = {1, -3, 3, 2, -2, -1, 1, 0, -1}; // left operand, 3x3 (with bias column)
    vector<float> Y_grad = {-2, -1, 3,  2, -5, 0,  -2, 5, -8,  3, -3, 1}; // dL/dy
    vector<float> D_data = {10, -5, 17, 14, -41, -6, 15, 11}; // expected result dL/dx, 2x4
    Tensor W(3, 3);
    Tensor Y(4, 3);
    Tensor DmatDx(4, 2);
    cudaMemcpy(W.data, W_data.data(), 3*3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y.data, Y_grad.data(), 4*3 * sizeof(float), cudaMemcpyHostToDevice);

    matmulDerived<<<1, {16, 16}>>>(DmatDx, Y, W);
    vector<float> D_result = static_cast<vector<float>>(DmatDx);
    bool pass = true;
    for(size_t i = 0; i < D_data.size(); i++)
      if(fabs(D_data[i] - D_result[i]) > 0.0001)
        pass = false;

    cout << (pass ? "pass" : "fail") << "\n";

    if(!pass) {
      Tensor D_expected(4,2);
      cudaMemcpy(D_expected.data, D_data.data(), 4*2 * sizeof(float), cudaMemcpyHostToDevice);
      D_expected.print(std::cout, "matmulDerived expected");
      DmatDx.print(std::cout, "matmulDerived result");
    }
  }

  {
    // vector<float> W_data = {1, -3, 3, 2, -2, -1}; // expected result, 3x2
    // Tensor DmatDx({3, 2});
    // dmatmul_dx(A, DmatDx);
    // vector<float> D_result = DmatDx;
    // pass = true;
    // for(size_t i = 0; i < D_data.size(); i++)
    //   if(fabs(D_data[i] - D_result[i]) > 0.0001)
    //     pass = false;
    // cout << "  backward dx: " << (pass ? "pass" : "fail") << "\n";
  }
}
}
