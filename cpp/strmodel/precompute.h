#include <memory>
#include <string>
#include "dataio/numpywrite.h"
#include "neuralnet/nninterface.h"

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

  int count; // first dimension size; rows entered
  int capacity; // first dimension size; max rows
  std::unique_ptr<NumpyBuffer<int>> movepos; // next move locations
  std::unique_ptr<NumpyBuffer<float>> trunkOutputNCHW; // NN state after last batch norm, without heads

  // extract input tensor and add it as new row
  void addBoard(Board& board, const BoardHistory& history, Move move);
  // run all added boards through the neural net
  void evaluate();
  // reset the extractor
  void clear();
  void selectIndex(int index); // move the data from the specified index to the first position, then set count=1
  int readFeaturesFromSgf(const std::string& filePath, Player pla = 0);
  void readFeaturesFromZip(const std::string& filePath);
  // write trunk & loc features in binary format
  void writeFeaturesToZip(const std::string& filePath);
  void writeInputsToNpz(const std::string& filePath);
  void writeOutputsToNpz(const std::string& filePath);
  void writePicksToNpz(const std::string& filePath);

  constexpr static int maxBatchSize = 1000;
  constexpr static int nnXLen = 19;
  constexpr static int nnYLen = 19;
  constexpr static int numTrunkFeatures = 384;  // strength model is limited to this size
  constexpr static int trunkSize = nnXLen*nnYLen*numTrunkFeatures;
  constexpr static int gpuIdx = -1;

private:

  void allocateBuffers();

  std::unique_ptr<ComputeHandle, ComputeHandleDeleter> handle;
  std::unique_ptr<InputBuffers, InputBuffersDeleter> inputBuffers;

  int numSpatialFeatures;
  int numGlobalFeatures;

};


#endif
