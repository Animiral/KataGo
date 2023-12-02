// Strength network addon

#ifndef NEURALNET_STRENGTHNET_H_
#define NEURALNET_STRENGTHNET_H_

#include "cudaincludes.h"
#include "core/rand.h"
#include <vector>

// C++ wrapper over CUDA
struct Tensor {

  float* data; // GPU device pointer; column-major order
  uint2 dims;

  Tensor() = default;
  explicit Tensor(std::vector<float> data_, uint2 dims_); // host to GPU
  explicit Tensor(uint2 dims_);
  Tensor(const Tensor& rhs);                    // copy without ownership (for passing as arg to kernel)
  Tensor(Tensor&& rhs) noexcept;
  Tensor& operator=(const Tensor& rhs);         // ownership remains with rhs; use clone() for a full copy
  Tensor& operator=(Tensor&& rhs) noexcept;
  // Tensor& operator=(std::vector<float> data_);
  ~Tensor() noexcept;

  explicit operator std::vector<float>() const; // GPU to host
  void randomInit(Rand& rand);                  // new weights
  Tensor clone() const;                         // copy with ownership
  float variance() const;
  void print(std::ostream& stream, const std::string& name);

private:

  bool isOwner; // this Tensor has ownership of data ptr

};

Tensor& matmul(const Tensor& weights, const Tensor& x, Tensor& y);
Tensor& dmatmul_dx(const Tensor& weights, Tensor& j);
Tensor& dmatmul_dweights(const Tensor& weights, const Tensor& x, Tensor& j);
Tensor& relu(Tensor& x);
Tensor& drelu_dx(const Tensor& x, Tensor& j);
Tensor& tanh_scale_softmax(Tensor& x, float scale);
Tensor& dtanh_scale_softmax_dx(const Tensor& x, float scale, Tensor& j);
Tensor& weighed_average(Tensor& x, const Tensor& weights);
Tensor& dweighed_average_dx(const Tensor& weights, Tensor& j);

// this is what we give as input to the strength model for a single move
struct MoveFeatures {
  float winProb;
  float lead;
  float movePolicy;
  float maxPolicy;
  float winrateLoss;  // compared to previous move
  float pointsLoss;  // compared to previous move
};

Tensor makeInputTensor(std::vector<MoveFeatures> features);
Tensor makeOutputTensor(float target); // for backwards pass; scale target to NN output ((y-1500)/500)
float scaleOutputTensor(const Tensor& output); // includes scaling from NN output * 500 + 1500

// Implements the strength network.
// Currently this is a feed-forward network with one hidden layer which takes MoveFeatures as input
// and produces two outputs: the estimated player rating and the log-contribution of the data point
// to the aggregated rating.
class StrengthNet {

public:

  using Input = std::vector<MoveFeatures>;
  using Output = float;

  StrengthNet(); // weights are not initialized by default
  ~StrengthNet();

  void randomInit(Rand& rand);                 // new weights
  bool loadModelFile(const std::string& path); // load weights
  void saveModelFile(const std::string& path); // store weights

  Tensor& forward(const Tensor& input);
  void backward(const Tensor& target, float learnrate);  // buffers must be filled by forward pass

private:

  static const uint32_t STRNET_HEADER;
  static constexpr std::size_t maxN = 1000; // max number of input moves (for workspace buffer)
  static constexpr std::size_t in_ch = 6;
  static constexpr std::size_t hidden_ch = 32;
  static constexpr std::size_t out_ch = 2;
  std::size_t N; // last seen N in forward(), remember for backward()

  // all of these are device (GPU) pointers
  float* inputx;   // memory holding input features
  float* hiddenx;  // memory holding hidden features
  Tensor outputx;  // memory holding output features
  Tensor softx;    // memory holding softmaxed output features
  Tensor aggregx;  // memory holding output aggregate
  float* back1x;   // memory holding gradients 1 for backpropagation (double buffer)
  float* back2x;   // memory holding gradients 2 for backpropagation (double buffer)
  float* Wh;  // hidden layer weights
  float* bh;  // hidden layer bias
  float* Wo;  // output layer weights
  float* bo;  // output layer bias

};

#endif  // NEURALNET_STRENGTHNET_H_
