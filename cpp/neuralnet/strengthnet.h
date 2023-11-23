// Strength network addon

#ifndef NEURALNET_STRENGTHNET_H_
#define NEURALNET_STRENGTHNET_H_

#include "cudaincludes.h"
#include "core/rand.h"
#include <vector>

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
  ~StrengthNet();

  void randomInit(Rand& rand);                 // new weights
  bool loadModelFile(const std::string& path); // load weights
  void saveModelFile(const std::string& path); // store weights

  Output forward(const Input& input);
  void backward(Output target, float learnrate);  // buffers must be filled by forward pass

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
  float* outputx;  // memory holding output features
  float* softx;    // memory holding softmaxed output features
  float* aggregx;  // memory holding output aggregate
  float* back1x;   // memory holding gradients 1 for backpropagation (double buffer)
  float* back2x;   // memory holding gradients 2 for backpropagation (double buffer)
  float* Wh;  // hidden layer weights
  float* bh;  // hidden layer bias
  float* Wo;  // output layer weights
  float* bo;  // output layer bias

};

#endif  // NEURALNET_STRENGTHNET_H_
