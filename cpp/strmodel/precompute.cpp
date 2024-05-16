#include "precompute.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <zip.h>
#include "dataio/sgf.h"
#include "dataio/numpywrite.h"
#include "neuralnet/modelversion.h"

#include <iostream>

using std::unique_ptr;
using std::string;
using std::vector;
using namespace std::literals;

namespace {

unique_ptr<ComputeHandle, ComputeHandleDeleter> createComputeHandle(LoadedModel& loadedModel, int capacity);

}

PrecomputeFeatures::PrecomputeFeatures(LoadedModel& loadedModel, int cap)
: PrecomputeFeatures(cap)
{
  handle = createComputeHandle(loadedModel, cap);
  int modelVersion = NeuralNet::getModelVersion(&loadedModel);
  numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  inputBuffers.reset(NeuralNet::createInputBuffers(&loadedModel, capacity, PrecomputeFeatures::nnXLen, PrecomputeFeatures::nnYLen));
  allocateBuffers();
}

PrecomputeFeatures::PrecomputeFeatures(int cap)
: handle(nullptr, {}), inputBuffers(nullptr, {}),
  count(0),
  capacity(cap)
{
  numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(NNModelVersion::defaultModelVersion); // 22 features
  numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(NNModelVersion::defaultModelVersion); // 19 features
  allocateBuffers();
}

void PrecomputeFeatures::startGame(const std::string& sgfPath) {
  nextResult.sgfPath = sgfPath;
  nextResult.startIndex = 0;
  nextResult.trunk = trunk.data() + count * trunkSize;
  nextResult.movepos = movepos.data() + count;
  nextResult.player = plas.data() + count;
}

void PrecomputeFeatures::addBoard(Board& board, const BoardHistory& history, Move move) {
  assert(!isFull());
  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;
  bool inputsUseNHWC = false;
  // write to row
  float* binaryInput = NeuralNet::getSpatialBuffer(inputBuffers.get()) + count * nnXLen * nnYLen * numSpatialFeatures;
  float* globalInput = NeuralNet::getGlobalBuffer(inputBuffers.get()) + count * numGlobalFeatures;
  NNInputs::fillRowV7(board, history, move.pla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, binaryInput, globalInput);
  movepos[count] = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
  plas[count] = move.pla;
  count++;
}

void PrecomputeFeatures::endGame() {
  nextResult.endIndex = nextResult.startIndex + count - resultTip;
  results.push_back(nextResult);
  resultTip = count;
  nextResult = {};
}

bool PrecomputeFeatures::isFull() const {
  return count >= capacity;
}

std::vector<PrecomputeFeatures::Result> PrecomputeFeatures::evaluate() {
  if(count > 0) {
    NeuralNet::getOutputTrunk(handle.get(), inputBuffers.get(), count, trunk.data());
    // if there is an open game, it becomes a partial result; the process resembles endGame(), then startGame()
    if(count > resultTip) {
      nextResult.endIndex = nextResult.startIndex + count - resultTip;
      results.push_back(nextResult);
      nextResult.startIndex = nextResult.endIndex;
      nextResult.trunk = trunk.data();
      nextResult.movepos = movepos.data();
      nextResult.player = plas.data();
    }
  }
  count = resultTip = 0;
  return move(results);
}

void PrecomputeFeatures::writeResultToMoveset(Result result, SelectedMoves::Moveset& moveset) {
  assert(result.startIndex <= result.endIndex);
  assert(result.endIndex <= moveset.moves.size());

  for(size_t i = result.startIndex; i < result.endIndex; i++) {
    float* trunkBegin = result.trunk + i*trunkSize;
    float* trunkEnd = result.trunk + (i+1)*trunkSize;
    moveset.moves[i].trunk.reset(new vector<float>(trunkBegin, trunkEnd));
    moveset.moves[i].pos = result.movepos[i];
  }
}

void PrecomputeFeatures::writeInputsToNpz(const string& filePath) {
  int rows = count;
  auto binaryInputNCHW = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows, numSpatialFeatures, nnXLen, nnYLen});
  auto globalInputNC = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows, numGlobalFeatures});
  std::copy_n(NeuralNet::getSpatialBuffer(inputBuffers.get()), rows*nnXLen*nnYLen*numSpatialFeatures, binaryInputNCHW->data);
  std::copy_n(NeuralNet::getGlobalBuffer(inputBuffers.get()), rows*numGlobalFeatures, globalInputNC->data);
  auto moveposN = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows});
  std::copy_n(movepos.begin(), rows, moveposN->data);

  ZipFile zipFile(filePath);
  uint64_t numBytes = binaryInputNCHW->prepareHeaderWithNumRows(rows);
  zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW->dataIncludingHeader, numBytes);
  numBytes = moveposN->prepareHeaderWithNumRows(rows);
  zipFile.writeBuffer("movepos", moveposN->dataIncludingHeader, numBytes);
  numBytes = globalInputNC->prepareHeaderWithNumRows(rows);
  zipFile.writeBuffer("globalInputNC", globalInputNC->dataIncludingHeader, numBytes);
  zipFile.close();
}

void PrecomputeFeatures::writeOutputsToNpz(const string& filePath) {
  int rows = count;
  auto trunkOutputNCHW = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows, numTrunkFeatures, nnXLen, nnYLen});
  std::copy_n(trunk.begin(), rows*trunkSize, trunkOutputNCHW->data);
  auto moveposN = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows});
  std::copy_n(movepos.begin(), rows, moveposN->data);

  ZipFile zipFile(filePath);
  uint64_t numBytes = trunkOutputNCHW->prepareHeaderWithNumRows(rows);
  zipFile.writeBuffer("trunkOutputNCHW", trunkOutputNCHW->dataIncludingHeader, numBytes);
  numBytes = moveposN->prepareHeaderWithNumRows(rows);
  zipFile.writeBuffer("movepos", moveposN->dataIncludingHeader, numBytes);
  zipFile.close();
}

void PrecomputeFeatures::writePicksToNpz(const string& filePath) {
  int rows = count;
  NumpyBuffer<float> pickNC({rows, numTrunkFeatures});

  for(int i = 0; i < rows; i++) {
    int pos = movepos[i];
    if(pos >= 0 && pos < nnXLen * nnYLen) {
      for(int j = 0; j < numTrunkFeatures; j++) {
        pickNC.data[i*numTrunkFeatures + j] = trunk[i*trunkSize + j*nnXLen*nnYLen + pos];
      }
    }
    else {
      std::fill(pickNC.data + i*numTrunkFeatures, pickNC.data + (i+1)*numTrunkFeatures, 0);
    }
  }

  ZipFile zipFile(filePath);
  uint64_t numBytes = pickNC.prepareHeaderWithNumRows(rows);
  zipFile.writeBuffer("pickNC", pickNC.dataIncludingHeader, numBytes);
  zipFile.close();
}

void PrecomputeFeatures::allocateBuffers() {
  trunk.resize(capacity * trunkSize);
  movepos.resize(capacity);
  plas.resize(capacity);
  count = resultTip = 0;
}

void ComputeHandleDeleter::operator()(ComputeHandle* handle) noexcept {
  return NeuralNet::freeComputeHandle(handle);
}

void InputBuffersDeleter::operator()(InputBuffers* buffers) noexcept {
  return NeuralNet::freeInputBuffers(buffers);
}

namespace {

unique_ptr<ComputeHandle, ComputeHandleDeleter> createComputeHandle(LoadedModel& loadedModel, int capacity) {
  enabled_t useFP16Mode = enabled_t::False;
  enabled_t useNHWCMode = enabled_t::False;
  auto* computeContext = NeuralNet::createComputeContext(
    {PrecomputeFeatures::gpuIdx}, nullptr, PrecomputeFeatures::nnXLen, PrecomputeFeatures::nnYLen,
    "", "", false,
    useFP16Mode, useNHWCMode, &loadedModel
  );

  bool requireExactNNLen = true;
  bool inputsUseNHWC = false;
  unique_ptr<ComputeHandle, ComputeHandleDeleter> gpuHandle {
    NeuralNet::createComputeHandle(
      computeContext,
      &loadedModel,
      nullptr,
      capacity,
      requireExactNNLen,
      inputsUseNHWC,
      PrecomputeFeatures::gpuIdx,
      0
    ),
    ComputeHandleDeleter()
  };

  return gpuHandle;
}

}
