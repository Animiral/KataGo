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

  Tensor() = default;
  explicit Tensor(uint xdim, uint ydim, uint zdim = 1);
  Tensor(const Tensor& rhs);                    // copy without ownership (for passing as arg to kernel)
  Tensor(Tensor&& rhs) noexcept;
  Tensor& operator=(const Tensor& rhs);         // ownership remains with rhs; use clone() or copyFrom() for a full copy
  Tensor& operator=(Tensor&& rhs) noexcept;
  // Tensor& operator=(std::vector<float> data_);
  ~Tensor() noexcept;

  explicit operator std::vector<float>() const; // GPU to host
  void randomInit(Rand& rand);                  // new weights
  Tensor clone() const;                         // copy with ownership
  void assignFrom(const Tensor& rhs);           // same-size assign
  float variance() const;
  void print(std::ostream& stream, const std::string& name) const;

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
  void setBatchSize(size_t batchSize_) noexcept; // followed by forward&backward (batch_size times), then update()
  float getOutput() const;                     // GPU to host, with scaling
  void forward();
  void backward(float target, size_t index);   // buffers must be filled by forward pass
  void update(float weightPenalty, float learnrate);
  void printWeights(std::ostream& stream, const std::string& name) const;
  void printState(std::ostream& stream, const std::string& name) const;
  float thetaSq() const; // average of squared parameters (W1, W2r, W2z);
  float gradsSq() const; // average of squared parameter gradients (W1_grad, W2r_grad, W2z_grad);

private:

  static const uint32_t STRNET_HEADER;
  static constexpr std::size_t maxN = 1000; // max number of input moves (for workspace buffer)
  static constexpr std::size_t maxBatchSize = 100; // allocated 3rd dim of weight gradient tensors
  static constexpr std::size_t in_ch = 6;
  static constexpr std::size_t hidden_ch = 32;
  static constexpr std::size_t out_ch = 2;

  std::size_t N; // last seen N in forward(), remember for backward()
  std::size_t batchSize = 100; // allocated 3rd dim of weight gradient tensors

  Tensor x, h, r, a, y;  // (intermediate) calculation values
  Tensor h_grad, hr_grad, hz_grad, r_grad, z_grad, y_grad;  // intermediate gradients for backpropagation
  Tensor W1, W2r, W2z;  // parameters: weights with included biases
  Tensor W1_grad, W2r_grad, W2z_grad;  // parameter update gradients

};

#endif  // NEURALNET_STRENGTHNET_H_
