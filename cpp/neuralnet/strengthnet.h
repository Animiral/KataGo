// Strength network addon

#ifndef NEURALNET_STRENGTHNET_H_
#define NEURALNET_STRENGTHNET_H_

#include "cudaincludes.h"
#include "core/rand.h"
#include <vector>

// C++ wrapper over CUDA
struct Tensor {

  float* data; // GPU device pointer; column-major order
  uint3 dims;
  uint3 viewDims; // for broadcasting data along some dimensions
  bool transposed; // if true, data is considered in row-major order instead of column-major

  Tensor() = default;
  explicit Tensor(uint xdim, uint ydim, uint zdim = 1);
  Tensor(const Tensor& rhs);                    // copy without ownership (for passing as arg to kernel)
  Tensor(Tensor&& rhs) noexcept;
  Tensor& operator=(const Tensor& rhs) = delete;
  Tensor& operator=(Tensor&& rhs) = delete;
  ~Tensor() noexcept;

  explicit operator std::vector<float>() const; // GPU to host
  void randomInit(Rand& rand);                  // new weights
  Tensor clone() const;                         // copy with ownership
  void assignFrom(const Tensor& rhs);           // same-size assign
  void reshape(uint xdim, uint ydim = 1, uint zdim = 1);
  void broadcast(uint xdim, uint ydim = 1, uint zdim = 1);
  void transpose();                             // swap dims.x & dims.y, flip transposed
  void print(std::ostream& stream, const std::string& name, bool humanReadable = true) const;

  static float mean(std::initializer_list<Tensor> ts);
  static float variance(std::initializer_list<Tensor> ts);

private:

  bool isOwner; // this Tensor has ownership of data ptr

};

// this is what we give as input to the strength model for a single move
struct MoveFeatures {
  float winProb;
  float lead;
  float movePolicy;
  float maxPolicy;
  float winrateLoss;  // compared to previous move
  float pointsLoss;  // compared to previous move
};

namespace Tests {
void runStrengthModelTests();}

// Implements the strength network.
// Currently this is a feed-forward network with one hidden layer which takes MoveFeatures as input
// and produces two outputs: the estimated player rating and the log-contribution of the data point
// to the aggregated rating.
class StrengthNet {

public:

  using Input = std::vector<MoveFeatures>;
  using Output = float;

  StrengthNet(); // weights are not initialized by default

  void randomInit(Rand& rand);                 // new weights
  bool loadModelFile(const std::string& path); // load weights
  void saveModelFile(const std::string& path); // store weights

  void setInput(const std::vector<MoveFeatures>& features); // host to GPU, with scaling
  void setBatchSize(size_t batchSize_) noexcept; // followed by forward&backward, then update()
  float getOutput() const;                     // GPU to host, with scaling
  void forward();
  void backward(float target);   // buffers must be filled by forward pass
  void mergeGrads();
  void update(float weightPenalty, float learnrate);
  void printWeights(std::ostream& stream, const std::string& name, bool humanReadable = true) const;
  void printState(std::ostream& stream, const std::string& name, bool humanReadable = true) const;
  void printGrads(std::ostream& stream, const std::string& name, bool humanReadable = true) const;
  float thetaVar() const; // variance of parameters (W1, W2r, W2z);
  float gradsVar() const; // variance of parameter gradients (W1_grad, W2r_grad, W2z_grad);

private:

  static const uint32_t STRNET_HEADER;
  static constexpr std::size_t maxN = 1000; // max number of input moves (for workspace buffer)
  static constexpr std::size_t maxBatchSize = 100; // allocated 3rd dim of weight gradient tensors
  static constexpr std::size_t in_ch = 6;
  static constexpr std::size_t hidden_ch = 1; // 32;
  static constexpr std::size_t out_ch = 2;

  std::size_t N; // last seen N in forward(), remember for backward()
  std::size_t batchSize = 100; // allocated 3rd dim of weight gradient tensors

  Tensor x, h, /*r, a,*/ y;  // (intermediate) calculation values
  Tensor h_grad, /*hr_grad, hz_grad, r_grad, z_grad,*/ y_grad, tgt;  // intermediate gradients for backpropagation
  Tensor W, b; // W1, W2r, W2z;  // parameters: weights with included biases
  Tensor W_grad, b_grad; // W1_grad, W2r_grad, W2z_grad;  // parameter update gradients

  friend void Tests::runStrengthModelTests();

};

#endif  // NEURALNET_STRENGTHNET_H_