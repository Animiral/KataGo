#include <memory>
#include <string>
#include <vector>
#include <queue>
#include "neuralnet/nninterface.h"
#include "strmodel/dataset.h"

#ifndef STRMODEL_PRECOMPUTE_H
#define STRMODEL_PRECOMPUTE_H

struct ComputeHandleDeleter {
  void operator()(ComputeHandle* handle) noexcept;
};

struct InputBuffersDeleter {
  void operator()(InputBuffers* buffers) noexcept;
};

class PrecomputeFeatures {

public:

  PrecomputeFeatures(LoadedModel& loadedModel, int cap);
  explicit PrecomputeFeatures(int cap);

  struct Result {
    std::string sgfPath;
    int moves; // number of moves evaluated
    float* trunk; // output for every position in the game
    int* movepos;
    Player* player; // player to move for every position
  };

  int count; // first dimension size; rows entered
  int capacity; // first dimension size; max rows
  int carrySize; // number of rows in partial game
  // buffers: [0,carrySize) "carry area", [carrySize,carrySize+capacity) "result area"
  std::vector<float> trunk; // each entry of length trunkSize 
  std::vector<int> movepos;
  std::vector<Player> plas;

  // extract input tensor and add it as new row
  void addBoard(Board& board, const BoardHistory& history, Move move);
  // signal the end of input, finalize result for the game with the given path
  void endGame(const std::string& sgfPath);
  // run all added boards through the neural net, carry over results for partial games
  void evaluate();
  // get evaluated data on one game, valid until next evaluate()
  Result nextResult();
  bool hasResult();
  // prepare the extractor for more input, preserve partial result rows in carry buffers
  void flip();

  // move the data from the specified index to the first position, then set count=1
  void selectIndex(int index);

  std::pair<Result, Result> splitBlackWhite(Result result);

  // copy trunk data to moveset; sizes must match
  static void writeResultToMoveset(Result result, SelectedMoves::Moveset& moveset);
  // write trunk & loc features in binary format
  static void writeResultToZip(Result result, const std::string& filePath);
  void writeInputsToNpz(const std::string& filePath);
  void writeOutputsToNpz(const std::string& filePath);
  void writePicksToNpz(const std::string& filePath);

  constexpr static int nnXLen = 19;
  constexpr static int nnYLen = 19;
  constexpr static int numTrunkFeatures = 384;  // strength model is limited to this size
  constexpr static int trunkSize = nnXLen*nnYLen*numTrunkFeatures;
  constexpr static int gpuIdx = -1;

private:

  void allocateBuffers();

  std::unique_ptr<ComputeHandle, ComputeHandleDeleter> handle;
  std::unique_ptr<InputBuffers, InputBuffersDeleter> inputBuffers;

  std::queue<Result> results;
  size_t resultTip; // index of next row after finalized results

  int numSpatialFeatures;
  int numGlobalFeatures;

};


#endif
