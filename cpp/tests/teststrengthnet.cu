// unit tests for strength net kernel functions in strengthnet.cu

#include <vector>
#include <sstream>
#include "neuralnet/strengthnet.h"

using namespace std;

namespace StrengthNetKernels
{

// definitions in strengthnet.cu
__global__ void scale(Tensor y, const Tensor w);
__global__ void add(Tensor y, const Tensor a, const Tensor b);
__global__ void dotproduct(Tensor y, const Tensor a, const Tensor b);
__global__ void matmul(Tensor y, const Tensor W, const Tensor x);
__global__ void transposeMatmul(Tensor y, const Tensor a, const Tensor b);
__global__ void relu(Tensor h);
__global__ void softmax(Tensor a);
__global__ void lossDerived(Tensor y_grad, float target, const Tensor y);
__global__ void softmaxDerived(Tensor z_grad, const Tensor a);
__global__ void matmulDerived(Tensor x_grad, const Tensor y_grad, const Tensor W);
__global__ void update(Tensor W, const Tensor W_grad, float learnrate);

}

using namespace StrengthNetKernels;

namespace Tests {
void runStrengthModelTests() {
  cout << "Running strength model tests" << endl;

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
      C_expected.print(cout, "matmul expected");
      C.print(cout, "matmul result");
    }
  }

  {
    cout << "- transposeMatmul: ";
    vector<float> A_data = {1, -3, 3, 2, -2, -1}; // left operand, 3x2
    vector<float> B_data = {7, 3, -11, 8, -10, 8, 4, 7}; // right operand, 4x2
    vector<float> C_data = {-13, -1, 31,  19, -25, 1,  -3, 25, -37,  22, -38, 17, 3, -5, 2}; // expected result, 5x3 (with bias column)
    Tensor A(2, 3);
    Tensor B(2, 4);
    Tensor C(5, 3);
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
      Tensor C_expected(5,3);
      cudaMemcpy(C_expected.data, C_data.data(), 5*3 * sizeof(float), cudaMemcpyHostToDevice);
      C_expected.print(cout, "transposeMatmul expected");
      C.print(cout, "transposeMatmul result");
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
      A_expected.print(cout, "softmax expected");
      A.print(cout, "softmax result");
    }
  }

  {
    cout << "- softmaxDerived: ";
    vector<float> A_data = {.3f, .5f, .2f}; // softmax outputs
    vector<float> Z_data = {1.f, 2.f, 3.f}; // gradient in & out
    vector<float> G_data = {-.27f, .05f, .22f}; // expected gradient out
    Tensor A(3, 1);
    Tensor Z(3, 1);
    cudaMemcpy(A.data, A_data.data(), 3*1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Z.data, Z_data.data(), 3*1 * sizeof(float), cudaMemcpyHostToDevice);

    softmaxDerived<<<1, 3>>>(Z, A);
    vector<float> Z_result = static_cast<vector<float>>(Z);
    bool pass = true;
    for(size_t i = 0; i < G_data.size(); i++)
      if(fabs(G_data[i] - Z_result[i]) > 0.0001)
        pass = false;

    cout << (pass ? "pass" : "fail") << "\n";
    if(!pass) {
      Tensor G_expected(3,1);
      cudaMemcpy(G_expected.data, G_data.data(), 3*1 * sizeof(float), cudaMemcpyHostToDevice);
      G_expected.print(cout, "softmaxDerived expected");
      Z.print(cout, "softmaxDerived result");
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
      D_expected.print(cout, "matmulDerived expected");
      DmatDx.print(cout, "matmulDerived result");
    }
  }

  {
    cout << "- add,relu: ";
    vector<float> hr_data = {3, 4, 5, 6, 7};
    vector<float> hz_data = {-8, 1, -2, -7, 0};
    vector<float> h_data = {0, 5, 3, 0, 7};
    size_t N = hr_data.size();
    Tensor hr(N, 1);
    Tensor hz(N, 1);
    Tensor h(N, 1);
    cudaMemcpy(hr.data, hr_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(hz.data, hz_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    add<<<1, N>>>(h, hr, hz);
    relu<<<1, N>>>(h);
    vector<float> h_result = static_cast<vector<float>>(h);
    bool pass = true;
    for(size_t i = 0; i < h_data.size(); i++)
      if(fabs(h_data[i] - h_result[i]) > 0.0001)
        pass = false;

    cout << (pass ? "pass" : "fail") << "\n";

    if(!pass) {
      Tensor h_expected(N, 1);
      cudaMemcpy(h_expected.data, h_data.data(), 5*1 * sizeof(float), cudaMemcpyHostToDevice);
      h_expected.print(cout, "add,relu expected");
      h.print(cout, "add,relu result");
    }
  }
}
}
