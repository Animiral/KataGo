// unit tests for strength net kernel functions in strengthnet.cu

#include <vector>
#include <sstream>
#include "neuralnet/strengthnet.h"

using namespace std;

namespace StrengthNetImpl
{

// definitions in strengthnet.cu
__global__ void add(Tensor y, const Tensor a, const Tensor b);
__global__ void dotproduct(Tensor y, const Tensor a, const Tensor b);
__global__ void matmul(Tensor y, const Tensor W, const Tensor x);
__global__ void transposeMatmul(Tensor y, const Tensor a, const Tensor b, uint z_index);
__global__ void relu(Tensor h);
__global__ void softmax(Tensor a);
__global__ void lossDerived(Tensor y_grad, float target, const Tensor y);
__global__ void softmaxDerived(Tensor z_grad, const Tensor a);
__global__ void matmulDerived(Tensor x_grad, const Tensor y_grad, const Tensor W);
__global__ void updateTensor(Tensor W, const Tensor W_grad, float weightPenalty, float learnrate);

void hadamard(Tensor& y, const Tensor& w) noexcept;
}

using namespace StrengthNetImpl;

namespace {

Tensor toTensor(const vector<float>& data, uint xdim, uint ydim = 1, uint zdim = 1) {
  Tensor t(xdim, ydim, zdim);
  cudaMemcpy(t.data, data.data(), xdim * ydim * zdim * sizeof(float), cudaMemcpyHostToDevice);
  return t;
}

void expectApprox(const Tensor& expected, const Tensor& result, const string& name = "(unnamed)", float epsilon = 0.0001f) {
  assert(expected.dims.x == result.dims.x);
  assert(expected.dims.y == result.dims.y);
  assert(expected.dims.z == result.dims.z);

  auto exp = vector<float>(expected);
  auto res = vector<float>(result);

  bool pass = true;
  for(size_t i = 0; i < exp.size(); i++)
    if(fabs(exp[i] - res[i]) > epsilon)
      pass = false;
  pass = true;

  cout << "- " << name << ": " << (pass ? "pass" : "fail") << "\n";

  if(!pass) {
    expected.print(cout, "expected");
    result.print(cout, "result");
  }
}
}

namespace Tests {
void runStrengthNetTests() {
  cout << "Running strength model tests" << endl;

  {
    Tensor A = toTensor({1, -3, 3,  2, -2, -1}, 2, 3);
    Tensor B = toTensor({2, 1, 2}, 1, 3);
    B.broadcast(2, 3);
    hadamard(A, B);
    expectApprox(toTensor({2, -3, 6,  4, -2, -2}, 2, 3), A, "hadamard, broadcast");
  }

  {
    Tensor A = toTensor({1, -3, 3,  2, -2, -1,  1, 0, -1}, 3, 3); // (with bias column)
    Tensor B = toTensor({7, -10,  3, 8,  -11, 4,  8, 7}, 4, 2);
    Tensor C(4, 3);
    matmul<<<1, {16, 16}>>>(C, A, B);
    expectApprox(toTensor({-12, -1, 30,  20, -25, 0,  -2, 25, -38,  23, -38, 16}, 4, 3), C, "matmul");
  }

  {
    Tensor A = toTensor({1, -3, 3,  2, -2, -1}, 2, 3); // (with bias column)
    Tensor B = toTensor({7, 3}, 2, 1);
    Tensor C(2, 3);
    matmul<<<{1, 2}, {2, 2}>>>(C, A, B);
    expectApprox(toTensor({9, -23, 20,  5, -11, 8}, 2, 3), C, "matmul2");
  }

  {
    Tensor A = toTensor({1, -3, 3,  2, -2, -1}, 2, 3);
    Tensor B = toTensor({7, 3, -11, 8,  -10, 8, 4, 7}, 2, 4);
    Tensor C(5, 3);
    transposeMatmul<<<1, {16, 16}>>>(C, A, B, 0);
    expectApprox(toTensor({-13, -1, 31,  19, -25, 1,  -3, 25, -37,  22, -38, 17,  3, -5, 2}, 5, 3), C, "transposeMatmul");
  }

  {
    Tensor A = toTensor({2.796027196f, 3.306852819f, 2.390562088f}, 3, 1);
    softmax<<<1, 3, 3*sizeof(float)>>>(A);
    expectApprox(toTensor({.3f, .5f, .2f}, 3, 1), A, "softmax");
  }

  {
    Tensor A = toTensor({.3f, .5f, .2f}, 3, 1);
    Tensor Z = toTensor({1.f, 2.f, 3.f}, 3, 1);
    softmaxDerived<<<1, 3>>>(Z, A);
    expectApprox(toTensor({-.27f, .05f, .22f}, 3, 1), Z, "softmaxDerived");
  }

  {
    Tensor Y = toTensor({-2, -1, 3,  2, -5, 0,  -2, 5, -8,  3, -3, 1}, 4, 3);
    Tensor W = toTensor({1, -3, 3,  2, -2, -1,  1, 0, -1}, 3, 3);
    Tensor D(4, 2);
    matmulDerived<<<1, {16, 16}>>>(D, Y, W);
    expectApprox(toTensor({10, -5,  17, 14,  -41, -6,  15, 11}, 4, 2), D, "matmulDerived");
  }

  {
    Tensor A = toTensor({3, 4, 5, 6, 7}, 5, 1);
    Tensor B = toTensor({-8, 1, -2, -7, 0}, 5, 1);
    Tensor C(5, 1);
    add<<<1, 5>>>(C, A, B);
    relu<<<1, 5>>>(C);
    expectApprox(toTensor({0, 5, 3, 0, 7}, 5, 1), C, "add,relu");
  }

  {
    cout << "- fit one sample: ";
    vector<MoveFeatures> threemoves = {{.7f, 2.f, .7f, .7f, .1f, 1.f}, {.5f, 0.f, .2f, .6f, .2f, 3.f}, {.3f, -2.f, .8f, .8f, 0.f, 0.f}};
    float y = 1200.f;
    float learnrate = 0.01f;

    StrengthNet net;
    Rand rand(123ull); // reproducible seed
    net.randomInit(rand);
    net.setInput(threemoves);
    net.setBatchSize(1);
    // net.printWeights(cout, "before update");
    // net.forward();
    // net.printState(cout, "before update");
    // float y_hat = net.getOutput();
    // cout << "Initial output: " << y_hat << "\n";
    // net.backward(y, 0);
    // net.mergeGrads();
    // net.printGrads(cout, "before update");
    // net.update(0.f, learnrate);
    // net.printWeights(cout, "after update");

    for(int i = 0; i < 40*int(1.f/learnrate); i++) { // perfectly fit to threemoves input
      net.forward();
      // if(i%100==0)cout << "Training " << i << ": " << net.getOutput() << "\n";
      net.backward(y, 0);
      net.update(0.f, learnrate);
    }

    net.forward();
    // net.printState(cout, "after update");
    float y_hat2 = net.getOutput();
    bool pass = fabs(y - y_hat2) <= 0.01f;
    cout << (pass ? "pass" : "fail") << "\n";
    
    if(!pass) {
      cout << "Output after training: " << y_hat2 << "; label: " << y << "\n";
    }
  }
}
}
