#include "../tests/tests.h"
#include "../neuralnet/strengthnet.h"

using namespace std;
using namespace TestCommon;

void Tests::runStrengthModelTests() {
  cout << "Running strength model tests" << endl;
  ostringstream out;

  {
    cout << "- matmul: ";
    vector<float> A_data = {1, -3, 3, 2, -2, -1, 1, 0, -1}; // left operand, 3x3 (with bias column)
    vector<float> B_data = {7, -10, 3, 8, -11, 4, 8, 7}; // right operand, 2x4
    vector<float> C_data = {-12, -1, 30,  20, -25, 0,  -2, 25, -38,  23, -38, 16}; // expected result, 3x4
    Tensor A(A_data, {3, 3});
    Tensor B(B_data, {4, 2});
    Tensor C({4, 3});

    matmul(A, B, C);
    vector<float> C_result = static_cast<vector<float>>(C);
    bool pass = true;
    for(size_t i = 0; i < C_data.size(); i++)
      if(fabs(C_data[i] - C_result[i]) > 0.0001)
        pass = false;

    cout << (pass ? "pass" : "fail") << "\n";

    if(!pass) {
      Tensor(C_data,{4,3}).print(std::cout, "matmul expected");
      C.print(std::cout, "matmul result");
    }
  }

  {
    cout << "- dmatmul_dx: ";
    vector<float> A_data = {1, -3, 3, 2, -2, -1, 1, 0, -1}; // left operand, 3x3 (with bias column)
    vector<float> D_data = {1, -3, 3, 2, -2, -1}; // expected result, 3x2
    Tensor A(A_data, {3, 3});
    Tensor DmatDx({2, 3});

    dmatmul_dx(A, DmatDx);
    vector<float> D_result = static_cast<vector<float>>(DmatDx);
    bool pass = true;
    for(size_t i = 0; i < D_data.size(); i++)
      if(fabs(D_data[i] - D_result[i]) > 0.0001)
        pass = false;

    cout << (pass ? "pass" : "fail") << "\n";

    if(!pass) {
      Tensor(D_data,{2,3}).print(std::cout, "dmatmul_dx expected");
      DmatDx.print(std::cout, "matmul result");
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
