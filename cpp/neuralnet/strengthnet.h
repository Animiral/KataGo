// Strength network addon

#ifndef NEURALNET_STRENGTHNET_H_
#define NEURALNET_STRENGTHNET_H_

#include "cudaincludes.h"
#include "core/rand.h"
#include <vector>

// C++ wrapper over a chunk of CUDA memory.
// A Tensor with xdim=5, ydim=2, batchSize=3, zoffset={0, 2, 4, 5} looks like this:
// + ---> xdim
// | z[0]     z[1]    z[2]  // zoffset
// |    1  4    -3  0    9
// v   -5  0    -2 -1    0
// ydim
struct Tensor {

  float* data;     // GPU device pointer; column-major order
  uint* zoffset;   // dims.z+1 monotone increasing indexes into data array
  uint3 dims;      // x: #elements in all batches, y: #features, z: #batch
  uint3 viewDims;  // for broadcasting data along some dimensions
  bool transposed; // if true, data is considered in row-major order instead of column-major

  explicit Tensor(uint xdim, uint ydim);
  explicit Tensor(uint xdim, uint ydim, const std::vector<uint>& zs);
  Tensor(const Tensor& rhs);                    // copy without ownership (for passing as arg to kernel)
  Tensor(Tensor&& rhs) noexcept;
  Tensor& operator=(const Tensor& rhs) = delete;
  Tensor& operator=(Tensor&& rhs) = delete;
  ~Tensor() noexcept;

  explicit operator std::vector<float>() const; // GPU to host
  void randomInit(Rand& rand);                  // new weights
  Tensor clone() const;                         // copy with ownership
  void assignFrom(const Tensor& rhs);           // same-size assign
  // void reshape(uint xdim, uint ydim = 1, uint batchSize = 1);
  void broadcast(uint xdim, uint ydim = 1, uint batchSize = 1);
  void transpose();                             // swap dims.x & dims.y, flip transposed
  void cat();                                   // treat the whole batch as single 2D tensor; hcat normally, vcat if transposed
  void uncat();                                 // go back to viewing the tensor as a batch
  void print(std::ostream& stream, const std::string& name, bool humanReadable = true) const;

  static float mean(std::initializer_list<Tensor> ts);
  static float variance(std::initializer_list<Tensor> ts);

private:

  Tensor() = default;

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
  void runStrengthNetTests();
}

// Implements the strength network.
// Currently this is a feed-forward network with one hidden layer which takes MoveFeatures as input
// and produces two outputs: the estimated player rating and the log-contribution of the data point
// to the aggregated rating.
class StrengthNet {

public:

  using Input = std::vector<std::vector<MoveFeatures>>; // batch of recent move sets
  using Output = std::vector<float>; // batch of modeled ratings

  StrengthNet(); // weights are not initialized by default
  ~StrengthNet();

  void randomInit(Rand& rand);                 // new weights
  void loadModelFile(const std::string& path); // load weights
  void saveModelFile(const std::string& path); // store weights

  void setInput(const Input& features); // host to GPU, with scaling
  Output getOutput() const;             // GPU to host, with scaling
  void forward();
  void setTarget(const Output& targets); // before backward(), targets size must match last forward batch size
  void backward();                       // buffers must be filled by forward pass
  void update(float weightPenalty, float learnrate);
  void printWeights(std::ostream& stream, const std::string& name, bool humanReadable = true) const;
  void printState(std::ostream& stream, const std::string& name, bool humanReadable = true) const;
  void printGrads(std::ostream& stream, const std::string& name, bool humanReadable = true) const;
  float thetaVar() const; // variance of parameters (W1, W2r, W2z);
  float gradsVar() const; // variance of parameter gradients (W1_grad, W2r_grad, W2z_grad);

private:

  static const uint32_t STRNET_HEADER;
  static constexpr uint in_ch = 6;
  static constexpr uint hidden_ch = 32;
  static constexpr uint out_ch = 2;

  std::size_t N; // total number of currently processed moves (across all batches)
  std::vector<uint> zoffset; // batching offsets common to all data tensors

  // all non-parameter tensors are allocated to appropriate size with setInput()
  Tensor *x, *h, *r, *a, *ra, *y;  // (intermediate) calculation values
  Tensor *h_grad, *hr_grad, *hz_grad, *r_grad, *z_grad, *a_grad, *y_grad, *tgt;  // intermediate gradients for backpropagation
  Tensor W1, b1, W2r, b2r, W2z, b2z;  // parameters: weights with included biases
  Tensor *W1_grad, *b1_grad, *W2r_grad, *b2r_grad, *W2z_grad, *b2z_grad;  // parameter update gradients

  void allocateTensors(); // build tensors according to N, batchSize and zoffset
  void freeTensors() noexcept;

  friend void Tests::runStrengthNetTests();

};

#endif  // NEURALNET_STRENGTHNET_H_
