// unit tests for strength net kernel functions in strengthnet.cu

#include <vector>
#include <sstream>
#include "neuralnet/strengthnet.h"

using namespace std;

namespace StrengthNetImpl {

float getelem(const Tensor &t, uint x, uint y = 0, uint z = 0); // tensor element access (slow/wasteful)
void scale(Tensor& y, float w);
void hadamard(Tensor& y, const Tensor& w);
void matmul(Tensor& y, const Tensor& W, const Tensor& x);
void add(Tensor& y, const Tensor& x);
void relu(Tensor& t);
void min(Tensor& y, const Tensor& x);
void minDerived(Tensor& x_grad, const Tensor& y_grad, const Tensor& x, const Tensor& y);
void sum(Tensor& y, const Tensor& x);
void softmax(Tensor& a);
void softmaxDerived(Tensor& a_grad, const Tensor& z_grad, const Tensor& a);

}

using namespace StrengthNetImpl;

namespace {

 // tensor element access (slow/wasteful) with CUDA error check
float checkgetelem(const Tensor &t, uint x, uint y = 0, uint z = 0);
// build a tensor from the data with the specified dimensions
Tensor toTensor(const vector<float>& data, uint xdim, uint ydim = 1, vector<uint> zoffset = {0, 0});
// print the test name and "pass" if the result matches expected everywhere within epsilon, "fail" otherwise
void expectApprox(const Tensor& expected, const Tensor& result, const string& name = "(unnamed)", float epsilon = 0.0001f);

}

namespace Tests {
void runStrengthNetTests() {
  cout << "Running strength model tests" << endl;

  {
    cout << "- access with batching: ";
    Tensor A = toTensor({11, 12, 21, 22, 101, 102}, 3, 2, vector<uint>{0, 2, 3});
    bool pass = true;

    pass &= 11 == checkgetelem(A, 0, 0, 0);
    pass &= 12 == checkgetelem(A, 0, 1, 0);
    pass &= 21 == checkgetelem(A, 1, 0, 0);
    pass &= 22 == checkgetelem(A, 1, 1, 0);
    pass &= 101 == checkgetelem(A, 0, 0, 1);
    pass &= 102 == checkgetelem(A, 0, 1, 1);

    // now with broadcast and transpose
    Tensor B = toTensor({3, 4}, 1, 2);
    B.transpose();
    B.broadcast(2, 2, 2);
    pass &= 3  == checkgetelem(B, 0, 0, 0);
    pass &= 3 == checkgetelem(B, 0, 1, 0);
    pass &= 4 == checkgetelem(B, 1, 0, 0);
    pass &= 4 == checkgetelem(B, 1, 1, 0);
    pass &= 3  == checkgetelem(B, 0, 0, 0);
    pass &= 3 == checkgetelem(B, 0, 1, 0);
    pass &= 4 == checkgetelem(B, 1, 0, 0);
    pass &= 4 == checkgetelem(B, 1, 1, 0);
    pass &= 3 == checkgetelem(B, 0, 0, 1);
    pass &= 3 == checkgetelem(B, 0, 1, 1);

    cout << (pass ? "pass" : "fail") << "\n";
  }

  {
    Tensor A = toTensor({1, -3, 3,  2, -2, -1}, 2, 3);
    Tensor B = toTensor({2, 1, 2}, 1, 3);
    B.broadcast(2, 3);
    hadamard(A, B);
    expectApprox(toTensor({2, -3, 6,  4, -2, -2}, 2, 3), A, "hadamard, broadcast");
  }

  {
    Tensor A = toTensor({1, -3, 3,  2, -2, -1}, 2, 3);
    Tensor B = toTensor({7, -10,  3, 8,  -11, 4,  8, 7}, 4, 2);
    Tensor C(4, 3);
    matmul(C, A, B);
    expectApprox(toTensor({-13, -1, 31,  19, -25, 1,  -3, 25, -37,  22, -38, 17}, 4, 3), C, "matmul");
  }

  {
    Tensor A = toTensor({7, 3,  2, 6}, 2, 2);
    Tensor B = toTensor({1, -3,  -1, 1,   -4, -1}, 3, 2, {0, 2, 3});
    Tensor C(3, 2, {0, 2, 3});
    B.cat(); C.cat();
    matmul(C, A, B);
    C.uncat();
    expectApprox(toTensor({1, -15,  -5, 3,   -30, -18}, 3, 2, {0, 2, 3}), C, "batch matmul, left-hand singular");
  }

  {
    Tensor A = toTensor({7, 3,  2, 6,   -1, -1}, 3, 2, {0, 2, 3});
    Tensor B = toTensor({1, -3,  -1, 1,   -4, -1}, 3, 2, {0, 2, 3});
    Tensor C(2, 2, {0, 2});
    B.transpose();
    A.cat(); B.cat(); C.cat();
    matmul(C, A, B);
    C.uncat();
    expectApprox(toTensor({9, 1, -18, -2}, 2, 2, {0, 2}), C, "batch matmul, left-hand multi, right-hand transposed");
  }

  {
    Tensor A = toTensor({1, -3, 3,  2, -2, -1}, 2, 3);
    Tensor B = toTensor({7, 3, -11, 8,  -10, 8, 4, 7}, 2, 4);
    Tensor C(4, 3);
    B.transpose();
    matmul(C, A, B);
    expectApprox(toTensor({-13, -1, 31,  19, -25, 1,  -3, 25, -37,  22, -38, 17}, 4, 3), C, "transpose, matmul");
  }

  {
    Tensor A = toTensor({2.796027196f, 3.306852819f, 2.390562088f, 1.796027196f, 1.390562088f, 2.083709268f, 0.697414907f}, 7, 1, {0, 3, 7});
    Tensor A_accumulate(7, 1, {0, 3, 7});
    softmax(A);
    expectApprox(toTensor({.3f, .5f, .2f,  .3, .2, .4, .1}, 7, 1, {0, 3, 7}), A, "softmax");
  }

  {
    Tensor A = toTensor({.3f, .5f, .2f,  .1f, .2f, .7f}, 6, 1, {0, 3, 6});
    Tensor Z_grad = toTensor({1.f, 2.f, 3.f,  -1.f, -2.f, -3.f}, 6, 1, {0, 3, 6});
    Tensor A_grad(6, 1, {0, 3, 6});
    softmaxDerived(A_grad, Z_grad, A);
    expectApprox(toTensor({-.27f, .05f, .22f,  .16f, .12f, -.28f}, 6, 1, {0, 3, 6}), A_grad, "softmaxDerived");
  }

  {
    // matmul in backpropagation: Y = output grads, W = weights, D = input grads
    Tensor Y = toTensor({-2, -1, 3,  2, -5, 0,  -2, 5, -8,  3, -3, 1}, 4, 3);
    // Tensor W = toTensor({1, -3, 3,  2, -2, -1,  1, 0, -1}, 3, 3);
    Tensor W = toTensor({1, -3, 3,  2, -2, -1}, 2, 3);
    Tensor D(4, 2);
    W.transpose();
    matmul(D, W, Y);
    expectApprox(toTensor({10, -5,  17, 14,  -41, -6,  15, 11}, 4, 2), D, "matmul (derived)");
  }

  {
    Tensor A = toTensor({3, 4, 5, 6, 7}, 5, 1);
    Tensor B = toTensor({-8, 1, -2, -7, 0}, 5, 1);
    add(A, B);
    relu(A);
    expectApprox(toTensor({0, 5, 3, 0, 7}, 5, 1), A, "add,relu");
  }

  {
    Tensor A = toTensor({3, 4, 5, 6, 7, 8}, 3, 2);
    Tensor B = toTensor({2, -1}, 1, 2);
    B.broadcast(3, 2);
    add(A, B);
    expectApprox(toTensor({5, 3, 7, 5, 9, 7}, 3, 2), A, "broadcast,add");
  }

  {
    Tensor A = toTensor({3, 4,  5, 6,  7, 8,   5, 4,  3, 2,  1, 0, 2, 2}, 7, 2, {0, 3, 3, 7});
    Tensor B(3, 2, {0, 1, 2, 3});
    sum(B, A);
    expectApprox(toTensor({15, 18,  0, 0,  11, 8}, 3, 2, {0, 1, 2, 3}), B, "sum"); // also test sum of nothing = 0
  }

  {
    cout << "- two-batch smoke test: ";
    vector<MoveFeatures> threemoves = {{.7f, 2.f, .7f, .7f, .1f, 1.f}, {.5f, 0.f, .2f, .6f, .2f, 3.f}, {.3f, -2.f, .8f, .8f, 0.f, 0.f}};
    vector<MoveFeatures> twomoves = {{.4f, 1.f, .3f, .4f, .05f, 1.5f}, {.5f, 0.5f, .25f, .65f, .2f, 3.f}};
    vector<vector<MoveFeatures>> features = {threemoves, twomoves};
    StrengthNet net;
    Rand rand(123ull); // reproducible seed
    net.randomInit(rand);
    net.setInput(features);
    net.forward();
    vector<float> y = net.getOutput();
    cout << "Output = {" << y[0] << "," << y[1] << "}\n";
    net.setTarget({1200.f, 1300.f});
    net.backward();
    // net.printState(cout, "smoke test");
    // net.printGrads(cout, "smoke test");
  }

  {
    cout << "- fit two batched samples: ";
    vector<MoveFeatures> threemoves = {{.7f, 2.f, .7f, .7f, .1f, 1.f}, {.5f, 0.f, .2f, .6f, .2f, 3.f}, {.3f, -2.f, .8f, .8f, 0.f, 0.f}};
    vector<MoveFeatures> twomoves = {{.4f, 1.f, .3f, .4f, .05f, 1.5f}, {.5f, .5f, .25f, .65f, .2f, 3.f}};
    vector<vector<MoveFeatures>> features = {threemoves, twomoves};
    vector<float> ys = {1200.f, 1300.f};
    float learnrate = .1f;

    StrengthNet net;
    Rand rand(123ull); // reproducible seed
    net.randomInit(rand);
    net.setInput(features);
    net.setTarget(ys);
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

    for(int i = 0; i < 1500*int(1.f/learnrate); i++) { // perfectly fit to threemoves input
      net.forward();
      // if(i%100==0) {
      //   vector<float> y_hat2 = net.getOutput();
      //   cout << "Training " << i << ": {" << y_hat2[0] << "," << y_hat2[1] << "}\n";
      // }
      net.backward();
      net.update(0.f, learnrate);
    }

    net.forward();
    // net.printState(cout, "after update");
    vector<float> y_hat2 = net.getOutput();
    bool pass = fabs(ys[0] - y_hat2[0]) + fabs(ys[1] - y_hat2[1]) <= 0.01f;
    cout << (pass ? "pass" : "fail") << "\n";
    
    if(!pass) {
      cout << "Output after training: {" << y_hat2[0] << "," << y_hat2[1] << "}; label: {" << ys[0] << "," << ys[1] << "}\n";
    }
  }

  if(0) // disabled for calculation time
  {
    cout << "- large inputs: ";
    size_t batchSize = 2200; // covers >2 blocks of 1024 threads each
    vector<vector<MoveFeatures>> batch(batchSize);
    vector<float> targets(batchSize);
    Rand rand(123ull); // reproducible seed
    StrengthNet net;
    net.randomInit(rand);

    for(size_t b = 0; b < batchSize; b++) {
      size_t n = 2000 + rand.nextUInt(500);
      batch[b].resize(n);
      for(MoveFeatures& mf : batch[b]) {
        mf.winProb = static_cast<float>(rand.nextDouble(0, 1));
        mf.lead = static_cast<float>(rand.nextGaussian());
        mf.movePolicy = static_cast<float>(rand.nextDouble(0, 1));
        mf.maxPolicy = static_cast<float>(rand.nextDouble(mf.movePolicy, 1));
        mf.winrateLoss = static_cast<float>(rand.nextDouble(-5, 80));
        mf.pointsLoss = static_cast<float>(rand.nextDouble(-1, 10));
      }
      targets[b] = static_cast<float>(rand.nextGaussian() * 300 + 1500);
    }
    vector<uint32_t> perm(batchSize);
    rand.fillShuffledUIntRange(batchSize, perm.data());

    // net outputs must follow the same permutation as the input
    net.setInput(batch);
    net.forward();
    vector<float> outputs = net.getOutput();
    // net gradients must be the same before/after permutation
    net.setTarget(targets);
    net.backward();
    vector<float> W_grad = static_cast<vector<float>>(*net.W_grad);
    vector<float> b_grad = static_cast<vector<float>>(*net.b_grad);

    vector<vector<MoveFeatures>> permBatch(batchSize);
    vector<float> permTargets(batchSize);
    for(size_t b = 0; b < batchSize; b++) {
      permBatch[b] = batch[perm[b]];
      permTargets[b] = targets[perm[b]];
    }
    net.setInput(permBatch);
    net.forward();
    vector<float> permOutputs = net.getOutput();
    net.setTarget(permTargets);
    net.backward();
    vector<float> permW_grad(*net.W_grad);
    vector<float> permb_grad(*net.b_grad);

    bool pass = true;
    for(size_t b = 0; b < batchSize; b++) {
      if(permOutputs[b] != outputs[perm[b]])
        pass = false;
    }
    for(size_t i = 0; i < W_grad.size(); i++) {
      if(fabs(1-permW_grad[i]/W_grad[i]) > 0.01) { // tolerate some deviation due to float ops order
        pass = false; cout << "permW_grad["<<i<<"]=" << permW_grad[i] << " != W_grad["<<i<<"]=" << W_grad[i] << "! ";
      }
    }
    for(size_t i = 0; i < b_grad.size(); i++) {
      if(fabs(1-permb_grad[i]/b_grad[i]) > 0.01) { // tolerate some deviation due to float ops order
        pass = false; cout << "permb_grad["<<i<<"]=" << permb_grad[i] << " != b_grad["<<i<<"]=" << b_grad[i] << "! ";
      }
    }
    cout << (pass ? "pass" : "fail") << "\n";
  }
}
}

namespace { // helper functions

float checkgetelem(const Tensor &t, uint x, uint y, uint z) {
  float out = getelem(t, x, y, z);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess) {
    cerr << "CUDA Error: " << cudaGetErrorString(status) << "\n";
    std::terminate();
  }
  return out;
}

Tensor toTensor(const vector<float>& data, uint xdim, uint ydim, vector<uint> zoffset) {
  assert(zoffset.size() >= 2);
  if(2 == zoffset.size())
    zoffset[1] = xdim;
  Tensor t(xdim, ydim, zoffset);
  cudaMemcpy(t.data, data.data(), xdim * ydim * sizeof(float), cudaMemcpyHostToDevice);
  return t;
}

void expectApprox(const Tensor& expected, const Tensor& result, const string& name, float epsilon) {
  assert(expected.dims.x == result.dims.x);
  assert(expected.dims.y == result.dims.y);
  assert(expected.dims.z == result.dims.z);

  auto exp = vector<float>(expected);
  auto res = vector<float>(result);

  bool pass = true;
  for(size_t i = 0; i < exp.size(); i++)
    if(fabs(exp[i] - res[i]) > epsilon)
      pass = false;

  cout << "- " << name << ": " << (pass ? "pass" : "fail") << "\n";

  if(!pass) {
    expected.print(cout, "expected");
    result.print(cout, "result");
  }
}

}
