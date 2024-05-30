#include <memory>
#include <string>
#include <vector>
#include <queue>
#include "neuralnet/nninterface.h"
#include "neuralnet/nneval.h"
#include "strmodel/dataset.h"

#ifndef STRMODEL_PRECOMPUTE_H
#define STRMODEL_PRECOMPUTE_H

// struct ComputeHandleDeleter {
//   void operator()(ComputeHandle* handle) noexcept;
// };

// struct InputBuffersDeleter {
//   void operator()(InputBuffers* buffers) noexcept;
// };

class PrecomputeFeatures {

public:

  PrecomputeFeatures(LoadedModel& loadedModel, int cap);
  explicit PrecomputeFeatures(int cap);

  struct ResultRow {
    Player pla; // who made the move
    int pos; // where the move happened
    float winProb;
    float lossProb; // not necessarily 1-winProb because no result is possible
    float expectedScore; // predicted score at end of game by NN
    float lead; // predicted bonus points to make game fair
    float movePolicy; // policy at move location
    float maxPolicy; // best move policy
    PickOutput pick; // trunk features at move location
  };

  struct Result {
    std::string sgfPath;
    size_t startIndex; // index (in moveset, not in game) of first result move
    std::vector<ResultRow> rows;
  };

  // signal the start of input for the specified game
  void startGame(const std::string& sgfPath);
  // extract input tensor and add it as new row
  void addBoard(Board& board, const BoardHistory& history, Move move);
  // signal the end of input, finalize result for the game with the given path
  void endGame();
  bool isFull() const;
  // run all added boards through the neural net; invalidate all previous results
  std::vector<Result> evaluate(); // picks and POC features
  // std::vector<Result> evaluateTrunks();
  // std::vector<Result> evaluatePicks();

  // copy trunk data to moveset; sizes must match
  static void writeResultToMoveset(Result result, SelectedMoves::Moveset& moveset);
  void writeInputsToNpz(const std::string& filePath);
  void writeOutputsToNpz(const std::string& filePath);
  void writePicksToNpz(const std::string& filePath);

  constexpr static int nnXLen = 19;
  constexpr static int nnYLen = 19;
  constexpr static int numTrunkFeatures = 384;  // strength model is limited to this size
  constexpr static int trunkSize = nnXLen*nnYLen*numTrunkFeatures;
  constexpr static int gpuIdx = -1;

private:

  // void allocateBuffers();

  // std::unique_ptr<ComputeHandle, ComputeHandleDeleter> handle;
  // std::unique_ptr<InputBuffers, InputBuffersDeleter> inputBuffers;

  size_t count; // input rows (boards) entered
  size_t capacity; // max rows

  NNEvaluator* evaluator;
  // capacity-sized buffers
  // std::vector<NNResultBuf*> buffers;
  // std::vector<NNOutput*>& outputs;
  // std::vector<float> trunk; // each entry of length trunkSize 
  // std::vector<int> movepos;
  // std::vector<float> pick; // each entry of length numTrunkFeatures
  // std::vector<Player> plas;

  Result nextResult; // next result in the making by adding boards
  size_t resultTip; // index of buffer row where nextResult starts
  std::vector<Result> results;

  int numSpatialFeatures;
  int numGlobalFeatures;

};


#endif
